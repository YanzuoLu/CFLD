#!/bin/bash

source $(dirname "${CONDA_PYTHON_EXE}")/activate CFLD
export CUDA_VISIBLE_DEVICES=$1
shift

python pose_transfer_train.py $@ \
    INPUT.ROOT_DIR ./fashion