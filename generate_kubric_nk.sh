#!/bin/bash 

RES=(
    "960x540"
)
NK=(
    "1k"
)

# Uncomment lines below to generate the dataset at other resolutions
# RES=(
#    "960x540"
#    "1920x1080"
#    "3840x2160"
#    "7680x4320"
# )
# NK=(
#    "1k"
#    "2k"
#    "4k"
#    "8k"
# )

# If you want to generate other annotations, besides the forward flow, then replace:
# --key_outputs rgba backward_flow forward_flow depth normal object_coordinates
# in the command below
#
# P.S. --key_outputs can also accept "segmentation". However, it seems like there is a bug
# when using the pypng package in the kubric docker image and I do not know how to fix it.

for i in $(seq -f "%03g" 0 29)
do
    for ((j = 0 ; j < ${#RES[@]} ; j++))
    do
        CMD="docker run
            --rm
            --interactive
            --user $(id -u):$(id -g)
            --volume "$(pwd):/kubric"
            kubricdockerhub/kubruntu
            /usr/bin/python3 
            challenges/movi/movi_def_worker.py
            --resolution ${RES[${j}]}
            --frame_end 21
            --job-dir output/kubric_nk/${NK[${j}]}/${i}
            --config configs_kubric_nk/config_${i}.json
	        --key_outputs rgba forward_flow
            ${@:1}"
        echo ${CMD}
        eval ${CMD}
    done
done
