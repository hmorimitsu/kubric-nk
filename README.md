# Kubric-NK

This repository is a fork of Kubric ([https://github.com/google-research/kubric](https://github.com/google-research/kubric)) containing code to generate the Kubric-NK dataset.

The Kubric-NK dataset was presented in our paper:

> DPFlow: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework. Henrique Morimitsu, Xiaobin Zhu, Roberto M. Cesar-Jr, Xiangyang Ji, and Xu-Cheng Yin. CVPR 2025

The code for the DPFlow optical flow model is available as part of PTLFlow at [https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/dpflow](https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/dpflow).

## Usage

There are two main ways of using Kubric-NK:

1. Generation: if you want to render the dataset samples by yourself, or modify the generation process to render different samples. Read the instructions at [1. Data Generation](#1-data-generation).
2. Download: if you just want to use Kubric-NK's samples to evaluate your method. Read the instructions at [2. Downloading](#2-downloading).

## 1. Data Generation

1. install [Docker](https://www.docker.com/).

2. Run the command:

```bash
bash generate_kubric_nk.sh
```

By default, this script will generate the optical flow samples for at 1k resolution only. You can open the script and follow the instructions there to generate other samples.

If you want to modify the generation process entirely, then you can modify the script [challenges/movi/movi_def_worker.py](challenges/movi/movi_def_worker.py).

You can also modify the configuration files in the folder `configs_kubric_nk` to change the scene conditions for each sequence.

## 2. Downloading

If you just want to use the Kubric-NK samples to evaluate a model, you can download them from one of the servers below.

### Online servers:

Google Drive: https://drive.google.com/drive/folders/1vSShkqyYwLJYX38iJP3kg6x-DMpCn0R6?usp=sharing

Baidu Cloud: https://pan.baidu.com/s/1sR_uX-yTMXORfLf_FU4opQ (password: `kbnk`)

**IMPORTANT:** Some files are HUGE and you only need to download a few of them, depending on the application. See the instructions below to know which files to download.

### Explanation of file contents

The files containing visual samples follow this naming pattern `<sample_type>_<resolution>.zip`.

The following `<sample_type>` are available:

- `rgba`: input images
- `config`: calibration data for calculating the groundtruth
- `backward_flow`
- `depth`
- `forward_flow`
- `normal`
- `object_coordinates`

The `rgba` and `config` files are required, while the others depend on the desired type of annotation for the task.

If you want to replicate the Kubric-NK evaluation for optical flow estimation, then follow these steps:

1. download all `rgba*`, `config*`, and `forward_flow*` files.
2. unzip them according to the following folder structure:
```
kubric-nk
├── 1k
|   ├── 000
|   |   ├── config.json
|   |   ├── forward_flow_00000.png
|   |   ├── forward_flow_00001.png
|   |   ├── ...
|   |   ├── rgba_00000.png
|   |   ├── ...
|   ├── 001
|   ...
├── 2k
...
```



## Citation

If you use Kubric-NK, please consider citing the following papers:

```
@InProceedings{Morimitsu2025DPFlow,
  author    = {Morimitsu, Henrique and Zhu, Xiaobin and Cesar-Jr., Roberto M. and Ji, Xiangyang and Yin, Xu-Cheng},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title     = {{DPFlow}: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework},
  year      = {2025},
}
```

```
@InProceedings{greff2021kubric,
    title = {Kubric: a scalable dataset generator}, 
    author = {Klaus Greff and Francois Belletti and Lucas Beyer and Carl Doersch and
              Yilun Du and Daniel Duckworth and David J Fleet and Dan Gnanapragasam and
              Florian Golemo and Charles Herrmann and Thomas Kipf and Abhijit Kundu and
              Dmitry Lagun and Issam Laradji and Hsueh-Ti (Derek) Liu and Henning Meyer and
              Yishu Miao and Derek Nowrouzezahrai and Cengiz Oztireli and Etienne Pot and
              Noha Radwan and Daniel Rebain and Sara Sabour and Mehdi S. M. Sajjadi and Matan Sela and
              Vincent Sitzmann and Austin Stone and Deqing Sun and Suhani Vora and Ziyu Wang and
              Tianhao Wu and Kwang Moo Yi and Fangcheng Zhong and Andrea Tagliasacchi},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022},
}
```


## Disclaimer

The code and assets are based on [Kubric](https://github.com/google-research/kubric). Kubric-NK, however, is not related to Google in any way.
