#
# Train residual networks from the paper "Conflicting Clusters". Can then be 
# evaluated using "evaluate.py". 
#

try:
    # If we are running in a multi node multi gpu setup. Otherwise run 
    # with tensorflow defaults
    import cluster_setup
except ImportError:
    pass

import gc
import io
import os
import time
import argparse
import json
import csv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from models.factory import create_model
from data.factory import load_dataset
from config import get_config
config = get_config()


def train(train_ds, test_ds, train_writer, test_writer, log_dir_run):
    
    # Initialize
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync

    # See https://www.tensorflow.org/tutorials/distribute/custom_training#training_loop
    with strategy.scope():
        model = create_model(config)
        radam=tfa.optimizers.RectifiedAdam(
            learning_rate=config.learning_rate, 
            epsilon=1e-6,
            weight_decay=1e-2
        )
        optimizer = tfa.optimizers.Lookahead(radam)

    train_ds = strategy.experimental_distribute_dataset(train_ds)
    test_ds = strategy.experimental_distribute_dataset(test_ds)

    with strategy.scope():
        loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(y, pred):
            per_example_loss = loss_fun(y, pred)
            loss = tf.nn.compute_average_loss(
                per_example_loss, 
                global_batch_size=config.batch_size)
            return loss

    with strategy.scope():
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def reset_train_metrics():
        train_accuracy.reset_states()
        train_loss.reset_states()
        
    def reset_test_metrics():
        test_accuracy.reset_states()
        test_loss.reset_states()
        
    reset_train_metrics()
    reset_test_metrics()

    with strategy.scope():
        def train_step(x, y):
            with tf.GradientTape() as tape:
                layers = model(x, training=True)
                pred = layers[-1]
                loss = compute_loss(y, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_accuracy.update_state(y, pred)
            train_loss.update_state(loss)

            return grads, layers

        def test_step(x, y):
            layers = model(x, training=False)
            pred = layers[-1]
            loss = compute_loss(y, pred)

            test_accuracy.update_state(y, pred)
            test_loss.update_state(loss)
            return layers

    with strategy.scope():
        @tf.function
        def distributed_train_step(x, y):
            per_replica_grads, per_replica_layers = strategy.experimental_run_v2(train_step, args=(x, y,))

            # Single GPU setup don't have replica values, but we still want to 
            # ensure also smaller GPU systems.
            grads = []
            for per_replica_grad in per_replica_grads:
                if config.num_gpus > 1:
                    grads.append(tf.concat(per_replica_grad.values, axis=0))
                else:
                    grads.append(per_replica_grad)

            layers = []
            for per_replica_layer in per_replica_layers:
                if config.num_gpus > 1:
                    layers.append(tf.concat(per_replica_layer.values, axis=0))
                else:
                    layers.append(per_replica_layer)
            return grads, layers
        
        @tf.function
        def distributed_test_step(x, y):
            per_replica_layers = strategy.experimental_run_v2(test_step, args=(x, y,))

            # Single GPU setup don't have replica values, but we still want to 
            # ensure also smaller GPU systems.
            layers = []
            for per_replica_layer in per_replica_layers:
                if config.num_gpus > 1:
                    layers.append(tf.concat(per_replica_layer.values, axis=0))
                else:
                    layers.append(per_replica_layer)
            return layers


        #
        # TRAINING LOOP
        #
        for epoch in range(config.epochs):
            is_last_epoch = epoch >= config.epochs-1
            print("", flush=True)
            model.save_weights("%s/ckpt-%d" % (log_dir_run, epoch))

            if epoch % 5 == 0 or is_last_epoch:
                # Test
                for x, y in test_ds:
                    start = time.time()
                    layers = distributed_test_step(x, y)

                with test_writer.as_default(): 
                    log_tensorboard("TEST", start, test_accuracy, 
                        epoch, test_loss, [], model, layers, x)
                    reset_test_metrics() 

            # Train, but not the very last epoch as its no more used...
            if not is_last_epoch:
                for x, y in train_ds:
                    start = time.time()
                    grads, layers = distributed_train_step(x, y)

                with train_writer.as_default(): 
                    log_tensorboard("TRAIN", start, train_accuracy, epoch,
                        train_loss, grads, model, layers, x)
                    reset_train_metrics()
                train_writer.flush()        
        

def log_tensorboard(name, start, accuracy, epoch, loss, 
                    grads, model, layers, x):
        
    accuracy_val = accuracy.result().numpy()
    loss_val = loss.result().numpy()

    print("[%s] Epoch %d (%d): Loss %.7f ; Accuracy %.4f; Time/step %.3f" % (
        name, epoch, epoch,  loss_val,
        accuracy_val, time.time() - start), flush=True)

    tf.summary.scalar("Accuracy", accuracy_val, step=epoch)
    tf.summary.scalar("Loss", loss_val, step=epoch)

    for l in range(len(grads)):
        l_name = model.trainable_variables[l].name
        tf.summary.histogram("Gradient/%s" % l_name, grads[l], step=epoch)

    for l in range(len(layers)):
        tf.summary.histogram("Value/%d" % l, layers[l], step=epoch)

    x = x.values[0] if config.num_gpus > 1 else x
    tf.summary.image("Input data", x, step=epoch, max_outputs=3,)


#
# M A I N
#
def main():
    global config

    print("\n\n####################", flush=True)
    print("# TRAIN %s" % config.log_dir, flush=True)
    print("####################\n", flush=True)
    
    train_csv = []
    test_csv = []
    for r in range(config.runs):

        log_dir_run = "%s/%d" % (config.log_dir, r)

        if os.path.exists(log_dir_run):
            print("(Warning) %s exists. Skip training." % (log_dir_run))
            continue
        
        train_writer = tf.summary.create_file_writer("%s/train" % log_dir_run)
        test_writer = tf.summary.create_file_writer("%s/test" % log_dir_run)

        # Load dataset
        train_ds, test_ds = load_dataset(config, augment=True)

        # Write hyperparameters
        with train_writer.as_default():
            params = vars(config)
            text = "|Parameter|Value|  \n"
            text += "|---------|-----|  \n"
            for val in params:
                text += "|%s|%s|  \n" % (val, params[val])
            tf.summary.text("Hyperparameters", text, step=0)
        train_writer.flush()
        
        # Train        
        train(train_ds, test_ds, train_writer, test_writer, log_dir_run)
        

if __name__ == '__main__':
    main()