#!/bin/sh
LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/cuda-10.2/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH

UDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES

ipython3 --pdb DeepMimic.py -- --arg_file $1
