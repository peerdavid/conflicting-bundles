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
matplotlib.use('Agg')

from models.factory import create_model
from data.factory import load_dataset
import conflicting_bundle as cb
from config import get_config
import shutil
config = get_config()


def evaluate(train_ds, test_ds, writer, log_dir_run):
    
    model = create_model(config)
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
    start = config.epochs-5 if config.last_epoch_only else 0   # Take last n as we measure the ema
    
    for epoch in range(start, config.epochs, 1):
        print("\nEvaluate epoch %d" % epoch, flush=True)
        ckpt_name = "%s/ckpt-%d" % (log_dir_run, epoch)

        if not os.path.exists(ckpt_name + ".index"):
            print("(Warning) Ckpt for epoch %d does not exist." % epoch)
            continue 

        model.load_weights(ckpt_name)
        is_last_epoch = epoch >= config.epochs-1

        # Measure conflicts 
        conflicts = cb.bundle_entropy(
            model, train_ds, 
            config.batch_size, config.learning_rate,
            config.num_classes, config.conflicting_samples_size, 
            config.all_conflict_layers)
        conflicts_int = conflicts_integral(conflicts_int, conflicts)
        print("Num. bundles: %.0f; Bundle entropy: %.5f" % \
                (conflicts[-1][0], conflicts[-1][1]), flush=True)

        # Tensorboard shows entropy at step t and csv file the 
        # normalized integral of the bundle entropy
        with writer.as_default():
            tf.summary.scalar("bundle/Num", conflicts[-1][0], step=epoch)
            tf.summary.scalar("bundle/Entropy", conflicts[-1][1], step=epoch)
        writer.flush()

        # "The test accuracy is averaged over the last 5 epochs to exclude outliers."
        if epoch > config.epochs - 5:
            for x, y in test_ds:
                start = time.time()
                test_step(x, y)

    print("Test-Accuracy: " + str(test_accuracy.result().numpy()))
    print("Test-Loss: " + str(test_loss.result().numpy()))

    conflicts_int = [[c[0].numpy(), c[1].numpy()] for c in conflicts_int]
    ret = []
    ret.extend(conflicts_int[-1])
    ret.append(test_loss.result().numpy())
    ret.append(test_accuracy.result().numpy())
    return ret, conflicts_int


#
# HELPER
#
def conflicts_integral(conflicts_int, conflicts):
    if conflicts_int == None:
        return [[c[0], c[1] / float(config.epochs)] for c in conflicts]
    
    assert len(conflicts_int) == len(conflicts), "Dimension of exponential moving average does not match."
    ret = []
    for c in range(len(conflicts)):
        ret.append([
            conflicts[c][0],
            conflicts_int[c][1] + (conflicts[c][1] / float(config.epochs))
        ])
    return ret


#
# M A I N
#
def main():
    global config
    
    print("\n\n####################", flush=True)
    print("# EVALUATE %s" % config.log_dir, flush=True)
    print("####################\n", flush=True)

    test_csv = []
    layers_csv = []
    for r in range(config.runs):
        log_dir_run = "%s/%d" % (config.log_dir, r)

        # Log some things
        if not os.path.exists(log_dir_run):
            print("(Warning) Path %s does not exist." % log_dir_run)
            continue

        # Clean old log    
        shutil.rmtree("%s/cc/" % log_dir_run, ignore_errors=True)
        
        # Create new log
        writer = tf.summary.create_file_writer("%s/cc" % log_dir_run)

        # Evaluate conflicts
        train_ds, test_ds = load_dataset(config, augment=False)
        with tf.device('/gpu:0'):
            test_csv_row, layers_csv_row = evaluate(train_ds, test_ds, writer, log_dir_run)
            test_csv.append(test_csv_row)
            layers_csv.append(layers_csv_row)
    
    with open("%s/test.csv" % (config.log_dir), "w") as out:
        writer = csv.writer(out)
        writer.writerow([
            "num_bundle", "bundle_entropy", "test_loss", "test_accuracy"
        ])
        writer.writerows(test_csv)
    
    with open("%s/conflicting_layers.csv" % (config.log_dir), "w") as out:
        writer = csv.writer(out)
        for layer in layers_csv:
            writer.writerows(layer)
        


if __name__ == '__main__':
    main()