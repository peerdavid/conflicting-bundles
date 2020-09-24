#!/bin/bash

#
# See README.md to setup environment correctly
#
source env/bin/activate


#
# Experiments FNN
#
#python3 train.py --dataset="mnist" --num_layers="5" --width_layers="50" --log_dir="fnn/50/5" --model="fnn" --epochs=10
python3 train.py --dataset="mnist" --num_layers="25" --width_layers="50" --log_dir="fnn/50/25" --model="fnn" --epochs=50
#python3 train.py --dataset="mnist" --num_layers="50" --width_layers="50" --log_dir="fnn/50/50" --model="fnn" --epochs=50

#...
#python3 evaluate.py --dataset="mnist" --log_dir="fnn/50/5"
python3 evaluate.py --dataset="mnist" --log_dir="fnn/50/25"
#python3 evaluate.py --dataset="mnist" --log_dir="fnn/50/50"


# #
# # Experiments VGG
# #
# python3 train.py --dataset="cifar" --num_layers="50" --log_dir="vgg/50" --model="vgg" --epochs=10
# python3 train.py --dataset="cifar" --num_layers="76" --log_dir="vgg/76" --model="vgg" --epochs=10
# #...
# python3 evaluate.py --dataset="cifar" --log_dir="vgg/50" 
# python3 evaluate.py --dataset="cifar" --log_dir="vgg/76" 


# #
# # Experiments ResNet
# #
# python3 train.py --dataset="cifar" --num_layers="50" --log_dir="resnet/50" --model="resnet" --epochs=10
# python3 train.py --dataset="cifar" --num_layers="76" --log_dir="resnet/76" --model="resnet" --epochs=10
# # ...
# python3 evaluate.py --dataset="cifar" --log_dir="resnet/50" 
# python3 evaluate.py --dataset="cifar" --log_dir="resnet/76" 


# #
# # AutoTune algorithm - 3 executions to get new pruning
# #
# python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/0" --epochs=50
# python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/1" --epochs=10
# python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/2" --epochs=10
# # ...
# python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/0"
# python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/1"
# python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/2"