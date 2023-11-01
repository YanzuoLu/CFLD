#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate MSC
export CUDA_VISIBLE_DEVICES=$1
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
shift

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --dynamo_backend "no" \
    --main_process_port $PORT \
    pose_transfer_train.py $@ \
    INPUT.ROOT_DIR $HOME/datasets