# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Henrique Morimitsu to generate the Kubric-NK dataset

from argparse import Namespace
import json
import logging
import pprint
import random

import bpy
import numpy as np


import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
from kubric.core import objects
from kubric.randomness import default_rng, random_rotation


# --- CLI arguments
def init_parser():
    parser = kb.ArgumentParser()
    parser.add_argument("--objects_split", choices=["train", "test"], default="train")
    # Configuration for the objects of the scene
    parser.add_argument(
        "--min_num_static_objects",
        type=int,
        default=10,
        help="minimum number of static (distractor) objects",
    )
    parser.add_argument(
        "--max_num_static_objects",
        type=int,
        default=20,
        help="maximum number of static (distractor) objects",
    )
    parser.add_argument(
        "--min_num_dynamic_objects",
        type=int,
        default=1,
        help="minimum number of dynamic (tossed) objects",
    )
    parser.add_argument(
        "--max_num_dynamic_objects",
        type=int,
        default=3,
        help="maximum number of dynamic (tossed) objects",
    )
    # Configuration for the floor and background
    parser.add_argument("--floor_friction", type=float, default=0.3)
    parser.add_argument("--floor_restitution", type=float, default=0.5)
    parser.add_argument(
        "--backgrounds_split", choices=["train", "test"], default="train"
    )

    parser.add_argument(
        "--camera",
        choices=["fixed_random", "linear_movement", "linear_movement_linear_lookat"],
        default="linear_movement",
    )
    parser.add_argument("--camera_inner_radius", type=float, default=8.0)
    parser.add_argument("--camera_outer_radius", type=float, default=12.0)
    parser.add_argument("--min_camera_movement", type=float, default=0.0)
    parser.add_argument("--max_camera_movement", type=float, default=4.0)
    parser.add_argument("--min_motion_blur", type=float, default=0.0)
    parser.add_argument("--max_motion_blur", type=float, default=0.0)
    parser.add_argument("--motion_blur", type=float, default=None)

    # Configuration for the source of the assets
    parser.add_argument(
        "--kubasic_assets",
        type=str,
        default="gs://kubric-public/assets/KuBasic/KuBasic.json",
    )
    parser.add_argument(
        "--hdri_assets",
        type=str,
        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json",
    )
    parser.add_argument(
        "--gso_assets", type=str, default="gs://kubric-public/assets/GSO/GSO.json"
    )
    parser.add_argument("--save_state", dest="save_state", action="store_true")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--assets_path", type=str, default=None)
    parser.add_argument(
        "--key_outputs",
        type=str,
        nargs="+",
        default=("rgba", "forward_flow"),
        choices=(
            "rgba",
            "backward_flow",
            "forward_flow",
            "depth",
            "normal",
            "object_coordinates",
            "segmentation",
        ),
    )
    parser.add_argument(
        "--static_spawn_region", type=float, nargs=6, default=(-7, -7, 0, 7, 7, 10)
    )
    parser.add_argument(
        "--dynamic_spawn_region", type=float, nargs=6, default=(-5, -5, 1, 5, 5, 5)
    )
    parser.add_argument(
        "--velocity_range",
        type=float,
        nargs=6,
        default=(-4.0, -4.0, 0.0, 4.0, 4.0, 0.0),
    )
    parser.add_argument("--random_flags", action="store_true")
    parser.set_defaults(save_state=False, frame_end=24, frame_rate=12, resolution=512)
    return parser


def get_linear_camera_motion_start_end(
    rng,
    movement_speed: float,
    inner_radius: float = 8.0,
    outer_radius: float = 12.0,
    z_offset: float = 0.1,
):
    """Sample a linear path which starts and ends within a half-sphere shell."""
    while True:
        camera_start = np.array(
            kb.sample_point_in_half_sphere_shell(inner_radius, outer_radius, z_offset)
        )
        direction = rng.rand(3) - 0.5
        movement = direction / np.linalg.norm(direction) * movement_speed
        camera_end = camera_start + movement
        if (
            inner_radius <= np.linalg.norm(camera_end) <= outer_radius
            and camera_end[2] > z_offset
        ):
            return camera_start, camera_end


def get_linear_lookat_motion_start_end(
    rng,
    inner_radius: float = 1.0,
    outer_radius: float = 4.0,
):
    """Sample a linear path which goes through the workspace center."""
    while True:
        # Sample a point near the workspace center that the path travels through
        camera_through = np.array(
            kb.sample_point_in_half_sphere_shell(0.0, inner_radius, 0.0)
        )
        while True:
            # Sample one endpoint of the trajectory
            camera_start = np.array(
                kb.sample_point_in_half_sphere_shell(0.0, outer_radius, 0.0)
            )
            if camera_start[-1] < inner_radius:
                break

        # Continue the trajectory beyond the point in the workspace center, so the
        # final path passes through that point.
        continuation = rng.rand(1) * 0.5
        camera_end = camera_through + continuation * (camera_through - camera_start)

        # Second point will probably be closer to the workspace center than the
        # first point.  Get extra augmentation by randomly swapping first and last.
        if rng.rand(1)[0] < 0.5:
            tmp = camera_start
            camera_start = camera_end
            camera_end = tmp
        return camera_start, camera_end


def position_sampler_return_args(region):
    region = np.array(region, dtype=np.float32)

    def _sampler(obj: objects.PhysicalObject, rng):
        obj.position = (0, 0, 0)  # reset position to origin
        effective_region = np.array(region) - obj.aabbox
        position = rng.uniform(*effective_region)
        obj.position = position
        return position

    return _sampler


def rotation_sampler_return_args(axis=None):
    def _sampler(obj: objects.PhysicalObject, rng):
        quaternion = random_rotation(axis=axis, rng=rng)
        obj.quaternion = quaternion
        return quaternion

    return _sampler


def move_until_no_overlap_return_args(
    asset,
    simulator,
    spawn_region=((-1, -1, -1), (1, 1, 1)),
    max_trials=100,
    rng=default_rng(),
):
    for _ in range(max_trials):
        rs = rotation_sampler_return_args()
        quaternion = rs(asset, rng)

        ps = position_sampler_return_args(spawn_region)
        position = ps(asset, rng)

        if not simulator.check_overlap(asset):
            return quaternion, position
    else:
        raise RuntimeError("Failed to place", asset)


def main(FLAGS):
    if FLAGS.config_path is None:
        FLAGS.static_spawn_region = [
            [FLAGS.static_spawn_region[i + j] for j in range(3)] for i in [0, 3]
        ]
        FLAGS.dynamic_spawn_region = [
            [FLAGS.dynamic_spawn_region[i + j] for j in range(3)] for i in [0, 3]
        ]
        FLAGS.velocity_range = [
            [FLAGS.velocity_range[i + j] for j in range(3)] for i in [0, 3]
        ]
        config_dict = {}
        config_dict["flags"] = vars(FLAGS)
        loaded_config = False
    else:
        job_dir = FLAGS.job_dir
        resolution = FLAGS.resolution
        motion_blur = FLAGS.motion_blur
        key_outputs = FLAGS.key_outputs
        with open(FLAGS.config_path, "r") as f:
            config_dict = json.load(f)
            config_dict["flags"]["config_path"] = FLAGS.config_path
            FLAGS = Namespace(**config_dict["flags"])
        FLAGS.job_dir = job_dir
        FLAGS.resolution = resolution
        if motion_blur is not None:
            config_dict["motion_blur"] = motion_blur
        FLAGS.key_outputs = key_outputs
        loaded_config = True

    if FLAGS.assets_path is None:
        assets_dict = {}
    else:
        with open(FLAGS.assets_path, "r") as f:
            assets_dict = json.load(f)

    # --- Common setups & resources
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

    logging.info(FLAGS.config_path)
    logging.info(loaded_config)
    if loaded_config:
        flags_string = pprint.pformat(config_dict, indent=2, width=100)
        logging.info("LOADED config:")
        logging.info(flags_string)

    if "motion_blur" not in config_dict:
        config_dict["motion_blur"] = rng.uniform(
            FLAGS.min_motion_blur, FLAGS.max_motion_blur
        )
        logging.info("INIT %s=%s", "motion_blur", str(config_dict["motion_blur"]))
    if config_dict["motion_blur"] > 0.0:
        logging.info(f"Using motion blur strength {config_dict['motion_blur']}")

    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(
        scene,
        scratch_dir,
        use_denoising=True,
        samples_per_pixel=64,
        motion_blur=config_dict["motion_blur"],
    )
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    # --- Populate the scene
    # background HDRI
    if "train_backgrounds" not in assets_dict:
        train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
        assets_dict["train_backgrounds"] = [str(v) for v in train_backgrounds]
        assets_dict["test_backgrounds"] = [str(v) for v in test_backgrounds]
        logging.info(
            "INIT %s=%s", "len(train_backgrounds)", str(len(train_backgrounds))
        )
        logging.info("INIT %s=%s", "len(test_backgrounds)", str(len(test_backgrounds)))

    if "hdri_id" not in config_dict:
        if FLAGS.backgrounds_split == "train":
            logging.info(
                "Choosing one of the %d training backgrounds...",
                len(assets_dict["train_backgrounds"]),
            )
            hdri_id = rng.choice(assets_dict["train_backgrounds"])
        else:
            logging.info(
                "Choosing one of the %d held-out backgrounds...",
                len(assets_dict["test_backgrounds"]),
            )
            hdri_id = rng.choice(assets_dict["test_backgrounds"])
        config_dict["hdri_id"] = str(hdri_id)
        logging.info("INIT %s=%s", "hdri_id", config_dict["hdri_id"])

    background_hdri = hdri_source.create(asset_id=config_dict["hdri_id"])
    # assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", config_dict["hdri_id"])
    scene.metadata["background"] = config_dict["hdri_id"]
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(
        asset_id="dome",
        name="dome",
        friction=1.0,
        restitution=0.0,
        static=True,
        background=True,
    )
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    # Camera
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=35.0, sensor_width=36)
    if FLAGS.camera == "fixed_random":
        if "camera_position" not in config_dict:
            config_dict["camera_position"] = kb.sample_point_in_half_sphere_shell(
                inner_radius=7.0, outer_radius=9.0, offset=0.1
            )
            logging.info(
                "INIT %s=%s", "camera_position", str(config_dict["camera_position"])
            )
        scene.camera.position = config_dict["camera_position"]
        scene.camera.look_at((0, 0, 0))
    elif (
        FLAGS.camera == "linear_movement"
        or FLAGS.camera == "linear_movement_linear_lookat"
    ):
        is_panning = FLAGS.camera == "linear_movement_linear_lookat"
        if "camera_start" not in config_dict:
            camera_start, camera_end = get_linear_camera_motion_start_end(
                rng,
                movement_speed=rng.uniform(
                    low=FLAGS.min_camera_movement, high=FLAGS.max_camera_movement
                ),
                inner_radius=FLAGS.camera_inner_radius,
                outer_radius=FLAGS.camera_outer_radius,
            )
            config_dict["camera_start"] = [float(v) for v in camera_start]
            config_dict["camera_end"] = [float(v) for v in camera_end]
            logging.info("INIT %s=%s", "camera_start", str(config_dict["camera_start"]))
            logging.info("INIT %s=%s", "camera_end", str(config_dict["camera_end"]))
        if is_panning:
            if "lookat_start" not in config_dict:
                lookat_start, lookat_end = get_linear_lookat_motion_start_end(rng)
                config_dict["lookat_start"] = [float(v) for v in lookat_start]
                config_dict["lookat_end"] = [float(v) for v in lookat_end]
                logging.info(
                    "INIT %s=%s", "lookat_start", str(config_dict["lookat_start"])
                )
                logging.info("INIT %s=%s", "lookat_end", str(config_dict["lookat_end"]))

        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames
        for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
            interp = (frame - FLAGS.frame_start + 1) / (
                FLAGS.frame_end - FLAGS.frame_start + 3
            )
            scene.camera.position = interp * np.array(config_dict["camera_start"]) + (
                1 - interp
            ) * np.array(config_dict["camera_end"])
            if is_panning:
                scene.camera.look_at(
                    interp * np.array(config_dict["lookat_start"])
                    + (1 - interp) * np.array(config_dict["lookat_end"])
                )
            else:
                scene.camera.look_at((0, 0, 0))
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)

    # ---- Object placement ----
    if "train_split" not in assets_dict:
        train_split, test_split = gso.get_test_split(fraction=0.1)
        assets_dict["train_split"] = [str(v) for v in train_split]
        assets_dict["test_split"] = [str(v) for v in test_split]
        logging.info("INIT %s=%s", "len(train_split)", str(len(train_split)))
        logging.info("INIT %s=%s", "len(test_split)", str(len(test_split)))

    if "active_split" not in config_dict:
        if FLAGS.objects_split == "train":
            logging.info("Choosing one of the %d training objects...", len(train_split))
            active_split = "train_split"
        else:
            logging.info("Choosing one of the %d held-out objects...", len(test_split))
            active_split = "test_split"
        config_dict["active_split"] = active_split
        logging.info("INIT %s=%s", "active_split", config_dict["active_split"])

    # add STATIC objects
    if "num_static_objects" not in config_dict:
        config_dict["num_static_objects"] = rng.randint(
            FLAGS.min_num_static_objects, FLAGS.max_num_static_objects + 1
        )
        logging.info(
            "INIT %s=%s", "num_static_objects", str(config_dict["num_static_objects"])
        )
    logging.info("Placing %d static objects:", config_dict["num_static_objects"])

    if "static_assets" not in config_dict:
        config_dict["static_assets"] = {}
    for i in range(config_dict["num_static_objects"]):
        i = str(i)
        if i not in config_dict["static_assets"]:
            config_dict["static_assets"][i] = {}
            config_dict["static_assets"][i]["id"] = str(
                rng.choice(assets_dict[config_dict["active_split"]])
            )
            config_dict["static_assets"][i]["scale"] = rng.uniform(0.75, 3.0)

            logging.info(
                "INIT %s:%s:%s=%s",
                "static_assets",
                i,
                "id",
                config_dict["static_assets"][i]["id"],
            )
            logging.info(
                "INIT %s:%s:%s=%s",
                "static_assets",
                i,
                "scale",
                str(config_dict["static_assets"][i]["scale"]),
            )

        obj = gso.create(asset_id=config_dict["static_assets"][i]["id"])
        assert isinstance(obj, kb.FileBasedObject)
        obj.scale = config_dict["static_assets"][i]["scale"] / np.max(
            obj.bounds[1] - obj.bounds[0]
        )
        obj.metadata["scale"] = config_dict["static_assets"][i]["scale"]
        scene += obj

        if "quaternion" not in config_dict["static_assets"][i]:
            quaternion, position = move_until_no_overlap_return_args(
                obj,
                simulator,
                spawn_region=config_dict["flags"]["static_spawn_region"],
                rng=rng,
            )
            config_dict["static_assets"][i]["quaternion"] = [
                float(v) for v in quaternion
            ]
            config_dict["static_assets"][i]["position"] = [float(v) for v in position]

            logging.info(
                "INIT %s:%s:%s=%s",
                "static_assets",
                i,
                "quaternion",
                str(config_dict["static_assets"][i]["quaternion"]),
            )
            logging.info(
                "INIT %s:%s:%s=%s",
                "static_assets",
                i,
                "position",
                str(config_dict["static_assets"][i]["position"]),
            )
        else:
            obj.quaternion = config_dict["static_assets"][i]["quaternion"]
            obj.position = np.array(config_dict["static_assets"][i]["position"])

        obj.friction = 1.0
        obj.restitution = 0.0
        obj.metadata["is_dynamic"] = False
        logging.info("    Added %s at %s", obj.asset_id, obj.position)

    logging.info("Running 100 frames of simulation to let static objects settle ...")
    _, _ = simulator.run(frame_start=-100, frame_end=0)

    # stop any objects that are still moving and reset friction / restitution
    for obj in scene.foreground_assets:
        if hasattr(obj, "velocity"):
            obj.velocity = (0.0, 0.0, 0.0)
            obj.friction = 0.5
            obj.restitution = 0.5

    dome.friction = FLAGS.floor_friction
    dome.restitution = FLAGS.floor_restitution

    # Add DYNAMIC objects
    if "num_dynamic_objects" not in config_dict:
        config_dict["num_dynamic_objects"] = rng.randint(
            FLAGS.min_num_dynamic_objects, FLAGS.max_num_dynamic_objects + 1
        )
        logging.info(
            "INIT %s=%s", "num_dynamic_objects", str(config_dict["num_dynamic_objects"])
        )
    logging.info("Placing %d dynamic objects:", config_dict["num_dynamic_objects"])

    if "dynamic_assets" not in config_dict:
        config_dict["dynamic_assets"] = {}
    for i in range(config_dict["num_dynamic_objects"]):
        i = str(i)
        if i not in config_dict["dynamic_assets"]:
            config_dict["dynamic_assets"][i] = {}
            config_dict["dynamic_assets"][i]["id"] = str(
                rng.choice(assets_dict[config_dict["active_split"]])
            )
            config_dict["dynamic_assets"][i]["scale"] = rng.uniform(0.75, 3.0)

            logging.info(
                "INIT %s:%s:%s=%s",
                "dynamic_assets",
                i,
                "id",
                config_dict["dynamic_assets"][i]["id"],
            )
            logging.info(
                "INIT %s:%s:%s=%s",
                "dynamic_assets",
                i,
                "scale",
                str(config_dict["dynamic_assets"][i]["scale"]),
            )

        obj = gso.create(asset_id=config_dict["dynamic_assets"][i]["id"])
        assert isinstance(obj, kb.FileBasedObject)
        obj.scale = config_dict["dynamic_assets"][i]["scale"] / np.max(
            obj.bounds[1] - obj.bounds[0]
        )
        obj.metadata["scale"] = config_dict["dynamic_assets"][i]["scale"]
        scene += obj
        if "quaternion" not in config_dict["dynamic_assets"][i]:
            quaternion, position = move_until_no_overlap_return_args(
                obj,
                simulator,
                spawn_region=config_dict["flags"]["dynamic_spawn_region"],
                rng=rng,
            )
            config_dict["dynamic_assets"][i]["quaternion"] = [
                float(v) for v in quaternion
            ]
            config_dict["dynamic_assets"][i]["position"] = [float(v) for v in position]

            config_dict["dynamic_assets"][i]["velocity"] = [
                float(v)
                for v in (
                    rng.uniform(*config_dict["flags"]["velocity_range"])
                    - [obj.position[0], obj.position[1], 0]
                )
            ]

            logging.info(
                "INIT %s:%s:%s=%s",
                "dynamic_assets",
                i,
                "quaternion",
                str(config_dict["dynamic_assets"][i]["quaternion"]),
            )
            logging.info(
                "INIT %s:%s:%s=%s",
                "dynamic_assets",
                i,
                "position",
                str(config_dict["dynamic_assets"][i]["position"]),
            )
            logging.info(
                "INIT %s:%s:%s=%s",
                "dynamic_assets",
                i,
                "velocity",
                str(config_dict["dynamic_assets"][i]["velocity"]),
            )
        else:
            obj.quaternion = config_dict["dynamic_assets"][i]["quaternion"]
            obj.position = np.array(config_dict["dynamic_assets"][i]["position"])

        obj.velocity = np.array(config_dict["dynamic_assets"][str(i)]["velocity"])
        obj.metadata["is_dynamic"] = True
        logging.info("    Added %s at %s", obj.asset_id, obj.position)

    if FLAGS.save_state:
        logging.info(
            "Saving the simulator state to '%s' prior to the simulation.",
            output_dir / "scene.bullet",
        )
        simulator.save_state(output_dir / "scene.bullet")

    # Run dynamic objects simulation
    logging.info("Running the simulation ...")
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)

    # --- Rendering
    if FLAGS.save_state:
        logging.info("Saving the renderer state to '%s' ", output_dir / "scene.blend")
        renderer.save_state(output_dir / "scene.blend")

    logging.info("Rendering the scene ...")
    data_stack = renderer.render()

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [
        asset
        for asset in scene.foreground_assets
        if np.max(asset.metadata["visibility"]) > 0
    ]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True,
    )

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"], scene.assets, visible_foreground_assets
    )
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    selected_data_stack = {}
    for k in FLAGS.key_outputs:
        selected_data_stack[k] = data_stack[k]

    # Save to image files
    kb.write_image_dict(selected_data_stack, output_dir)
    kb.post_processing.compute_bboxes(
        data_stack["segmentation"], visible_foreground_assets
    )

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(
        filename=output_dir / "metadata.json",
        data={
            "metadata": kb.get_scene_metadata(scene),
            "camera": kb.get_camera_info(scene.camera),
            "instances": kb.get_instance_info(scene, visible_foreground_assets),
        },
    )
    kb.write_json(
        filename=output_dir / "events.json",
        data={
            "collisions": kb.process_collisions(
                collisions, scene, assets_subset=visible_foreground_assets
            ),
        },
    )

    kb.write_json(
        filename=output_dir / "config.json",
        data=config_dict,
    )

    kb.write_json(
        filename=output_dir / "assets.json",
        data=assets_dict,
    )

    kb.done()


def gen_random_flags(FLAGS):
    FLAGS.frame_rate = int(np.random.choice([6, 12, 24]))
    FLAGS.step_rate = int(np.random.choice([120, 240, 480]))
    min_radius = 8
    radius_interval = 5
    FLAGS.camera_inner_radius = min_radius + random.random() * min_radius
    FLAGS.camera_outer_radius = FLAGS.camera_inner_radius + radius_interval
    outer_factor = FLAGS.camera_outer_radius - min_radius - radius_interval
    norm_outer_factor = 1 + outer_factor / 5
    FLAGS.min_num_static_objects = norm_outer_factor * 10
    FLAGS.max_num_static_objects = norm_outer_factor * 20
    FLAGS.min_num_dynamic_objects = norm_outer_factor * 1
    FLAGS.max_num_dynamic_objects = norm_outer_factor * 5
    FLAGS.camera = str(np.random.choice(
        ["fixed_random", "linear_movement", "linear_movement_linear_lookat"]
    ))
    FLAGS.min_camera_movement = 0.0
    FLAGS.max_camera_movement = 10.0
    FLAGS.static_spawn_region = [
        -7 * norm_outer_factor,
        -7 * norm_outer_factor,
        0.0,
        7 * norm_outer_factor,
        7 * norm_outer_factor,
        10 * norm_outer_factor,
    ]
    FLAGS.dynamic_spawn_region = [
        -5 * norm_outer_factor,
        -5 * norm_outer_factor,
        norm_outer_factor,
        5 * norm_outer_factor,
        5 * norm_outer_factor,
        5 * norm_outer_factor,
    ]
    FLAGS.velocity_range = [
        -4 * norm_outer_factor,
        -4 * norm_outer_factor,
        0.0,
        4 * norm_outer_factor,
        4 * norm_outer_factor,
        0.0,
    ]
    return FLAGS


if __name__ == "__main__":
    parser = init_parser()
    FLAGS = parser.parse_args()

    if FLAGS.random_flags:
        FLAGS = gen_random_flags(FLAGS)
        print(FLAGS)

    main(FLAGS)
