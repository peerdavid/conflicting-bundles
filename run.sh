#!/bin/bash

#
# See README.md to setup environment correctly
#
source env/bin/activate
python3 train.py --dataset="mnist" --num_layers="150" --width_layers="10" --log_dir="width-10" --model="fnn" --epochs=2
python3 evaluate.py --dataset="mnist" --log_dir="width-10"

# #
# # Experiments FNN
# #
# python3 train.py --dataset="mnist" --num_layers="50" --width_layers="10" --log_dir="fnn/10/50" --model="fnn" --epochs=10
# python3 train.py --dataset="mnist" --num_layers="50" --width_layers="25" --log_dir="fnn/25/50" --model="fnn" --epochs=10
# # ...
# python3 evaluate.py --dataset="mnist" --log_dir="fnn/10/50"
# python3 evaluate.py --dataset="mnist" --log_dir="fnn/25/50"


# #
# # Experiments VGG
# #
# python3 train.py --dataset="cifar" --num_layers="50" --log_dir="vgg/50" --model="vgg" --epochs=10
# python3 train.py --dataset="cifar" --num_layers="76" --log_dir="vgg/76" --model="vgg" --epochs=10
# python3 train.py --dataset="cifar" --num_layers="100" --log_dir="vgg/100" --model="vgg" --epochs=10
# #...
# python3 evaluate.py --dataset="cifar" --log_dir="vgg/50" 
# python3 evaluate.py --dataset="cifar" --log_dir="vgg/76" 


# #
# # Experiments ResNet
# #
# python3 train.py --dataset="cifar" --num_layers="50" --log_dir="resnet/50" --model="resnet" --epochs=10
# python3 train.py --dataset="cifar" --num_layers="76" --log_dir="resnet/76" --model="resnet" --epochs=10
# python3 train.py --dataset="cifar" --num_layers="100" --log_dir="resnet/100" --model="resnet" --epochs=10
# # ...
# python3 evaluate.py --dataset="cifar" --log_dir="resnet/50" 
# python3 evaluate.py --dataset="cifar" --log_dir="resnet/76" 


# #
# # AutoTune algorithm - 3 executions to get new pruning
# #
# python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/0" --epochs=120
# python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/1" --epochs=120
# python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/2" --epochs=120
# # ...
# python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/0"
# python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/1"
# python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/2"