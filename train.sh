#!/bin/bash
#mpiexec -n $1 CUDA_VISIBLE_DEVICES=0 python3 DeepMimic_Optimizer.py --arg_file $2

python3 testGPU.py
#mpiexec -n $1 python3 DeepMimic_Optimizer.py --arg_file $2

nohup mpiexec -n $1 python3 DeepMimic_Optimizer.py --arg_file $2 & tail nohup.out
