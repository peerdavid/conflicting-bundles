#!/bin/bash

python3 train.py --dataset="cifar" --num_layers="50" --log_dir="no_residuals/50" --use_residual="False"
python3 train.py --dataset="cifar" --num_layers="76" --log_dir="no_residuals/76" --use_residual="False"
python3 train.py --dataset="cifar" --num_layers="50" --log_dir="residuals/50" --use_residual="True"
python3 train.py --dataset="cifar" --num_layers="76" --log_dir="residuals/76" --use_residual="True"
python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/0" --epochs=10
python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/1" --epochs=10
python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/2" --epochs=10
python3 evaluate.py --dataset="cifar" --log_dir="no_residuals/50" 
python3 evaluate.py --dataset="cifar" --log_dir="no_residuals/76" 
python3 evaluate.py --dataset="cifar" --log_dir="residuals/50" 
python3 evaluate.py --dataset="cifar" --log_dir="residuals/76" 
python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/0"
python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/1"
python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/2"