#!/bin/sh
#LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/cuda-10.2/targets/x86_64-linux/lib/:/usr/local/cuda-10.2/:/usr/lib/x86_64-linux-gnu/
LD_LIBRARY_PATH=/home/nekokitty/dev/freeglut-3.2.1/build/lib/:/usr/local/lib/
export LD_LIBRARY_PATH

#ipython3 --pdb DeepMimic.py -- --arg_file $1
#nohup ipython3 --pdb DeepMimic.py -- --arg_file $1 & tail nohup.out
ipython3 --pdb MGSim.py -- --arg_file $1 

