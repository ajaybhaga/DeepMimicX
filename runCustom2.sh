#!/bin/sh
LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/cuda-10.2/targets/x86_64-linux/lib/:/usr/local/cuda-10.2/:/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH

nohup ipython3 --pdb DeepMimic.py -- --arg_file $1 & tail nohup.out
