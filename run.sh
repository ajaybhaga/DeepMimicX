#!/bin/sh
LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/cuda-10.1/lib64:/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH

ipython3 --pdb DeepMimic.py -- arg_file args/run_humanoid3d_spinkick_args.txt
