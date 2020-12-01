#
# Evaluate residual networks from the paper "Conflicting bundles".
#
try:
    # If we are running in a multi node multi gpu setup. Otherwise run 
    # with tensorflow defaults
    import cluster_setup
except ImportError:
    pass

import io
import os
import time
import argparse
import json
import csv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('Agg')

from models.factory import create_model
from data.factory import load_dataset
import conflicting_bundle as cb
from config import get_config
import shutil
config = get_config()



def run_conflicts(train_ds, log_dir_run):
    
    model = create_model(config)    
    epoch = config.epochs-1
    ckpt_name = "%s/ckpt-%d" % (log_dir_run, epoch)

    if not os.path.exists(ckpt_name + ".index"):
        print("(Warning) Ckpt for epoch %d does not exist." % epoch)
        exit(1) 

    model.load_weights(ckpt_name)

    # Measure conflicts 
    conflicts = cb.bundle_entropy(
        model, train_ds, 
        config.batch_size, config.learning_rate,
        config.num_classes, 
        evaluation_size=config.conflicting_samples_size, 
        all_layers=config.all_conflict_layers)
    return conflicts


def run_lesion(test_ds, log_dir_run, lesion):
    
    model = create_model(config, lesion)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(y, pred):
        per_example_loss = loss_fun(y, pred)
        loss = tf.nn.compute_average_loss(
            per_example_loss, 
            global_batch_size=config.batch_size)
        return loss

    @tf.function
    def test_step(x, y):
        pred = model(x, training=False)
        loss = compute_loss(y, pred)

        test_accuracy.update_state(y, pred)
        test_loss.update_state(loss)
        return pred

    #
    # Evaluation loop
    #
    conflicts_int = None    
    epoch = config.epochs-1
    ckpt_name = "%s/ckpt-%d" % (log_dir_run, epoch)

    if not os.path.exists(ckpt_name + ".index"):
        print("(Warning) Ckpt for epoch %d does not exist." % epoch)
        exit(1) 

    model.load_weights(ckpt_name)
    is_last_epoch = epoch >= config.epochs-1

    # Measure test accuracy
    for x, y in test_ds:
        start = time.time()
        test_step(x, y)

    acc = test_accuracy.result().numpy()
    return acc


#
# M A I N
#
def main():
    global config
    
    print("\n####################", flush=True)
    print("Measure conflicting layers", flush=True)

    log_dir_run = "%s/0" % (config.log_dir)

    # Study conflicts
    train_ds, test_ds = load_dataset(config, augment=False)
    conflicts = run_conflicts(train_ds, log_dir_run)
    
    with open("%s/conflicting_layers.csv" % (config.log_dir), "a") as out:
        writer = csv.writer(out)
        row = []
        #row.append(lesion)
        row.extend([c[1].numpy() for c in conflicts])
        writer.writerow(row)

    # Conflicting layers 
    conflicting_layers = []
    non_conflicting_layers = []
    for i in range(int(len(conflicts) / 2)):
        if conflicts[i*2][1] <= 0.0:
            non_conflicting_layers.append(i)
        else:
            conflicting_layers.append(i)

    # Lesion study using the information of conflicts
    print("\n####################", flush=True)
    print("Lesion with single layers", flush=True)
    for drop_layer in range(-1, 59):       
        # Evaluate conflicts with given lesion
        train_ds, test_ds = load_dataset(config, augment=False)
        acc = run_lesion(test_ds, log_dir_run, [drop_layer])
        
        print("Drop layer %.d | Accuracy %.3f" % (drop_layer, acc), flush=True)
        with open("%s/lesion_layerwise.csv" % (config.log_dir), "a") as out:
            writer = csv.writer(out)
            writer.writerow([drop_layer, acc])
    
    # Lesion study from 1 up to 15 drops using only conflicting layers
    print("\n####################", flush=True)
    print("Lesion of only conflicting layers", flush=True)#
    for num_layers_drop in range(1, len(conflicting_layers)):   
        accs = []
        for _ in range(15):
            # Shuffle our conflicting layers each time to get a new combination
            # that we drop
            random.shuffle(conflicting_layers)
            layers_to_drop = conflicting_layers[:num_layers_drop]
            
            # Evaluate conflicts with given lesion
            train_ds, test_ds = load_dataset(config, augment=False)
            acc = run_lesion(test_ds, log_dir_run, layers_to_drop)
            accs.append(acc)
        
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        with open("%s/lesion_conflicting.csv" % (config.log_dir), "a") as out:
            writer = csv.writer(out)
            writer.writerow([num_layers_drop, acc_mean, acc_std])
        print("Num. drops: %.d | Acc mean: %.3f | Acc std.: %.3f" % (num_layers_drop, acc, acc_std), flush=True)


    print("\n####################", flush=True)
    print("Lesion of only non-conflicting layers", flush=True)#
    for num_layers_drop in range(1, len(non_conflicting_layers)):   
        accs = []
        for _ in range(15):
            # Shuffle our conflicting layers each time to get a new combination
            # that we drop
            random.shuffle(non_conflicting_layers)
            layers_to_drop = non_conflicting_layers[:num_layers_drop]
            
            # Evaluate conflicts with given lesion
            train_ds, test_ds = load_dataset(config, augment=False)
            acc = run_lesion(test_ds, log_dir_run, layers_to_drop)
            accs.append(acc)
        
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        with open("%s/lesion_non_conflicting.csv" % (config.log_dir), "a") as out:
            writer = csv.writer(out)
            writer.writerow([num_layers_drop, acc_mean, acc_std])
        print("Num. drops: %.d | Acc mean: %.3f | Acc std.: %.3f" % (num_layers_drop, acc, acc_std), flush=True)


    print("\n####################", flush=True)
    print("Lesion of all layers", flush=True)
    all_layers = [i for i in range(0, 58)]
    for num_layers_drop in range(1, max(len(conflicting_layers), len(non_conflicting_layers))):   
        accs = []
        for _ in range(15):
            # Shuffle our conflicting layers each time to get a new combination
            # that we drop
            random.shuffle(all_layers)
            layers_to_drop = all_layers[:num_layers_drop]
            
            # Evaluate conflicts with given lesion
            train_ds, test_ds = load_dataset(config, augment=False)
            acc = run_lesion(test_ds, log_dir_run, layers_to_drop)
            accs.append(acc)
        
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        with open("%s/lesion_all.csv" % (config.log_dir), "a") as out:
            writer = csv.writer(out)
            writer.writerow([num_layers_drop, acc_mean, acc_std])
        print("Num. drops: %.d | Acc mean: %.3f | Acc std.: %.3f" % (num_layers_drop, acc_mean, acc_std), flush=True)



if __name__ == '__main__':
    main()