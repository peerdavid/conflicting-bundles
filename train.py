#
# Train residual networks from the paper "Conflicting bundles". Can then be 
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
from tensorflow.keras.mixed_precision import experimental as mixed_precision


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
                pred = model(x, training=True)
                loss = compute_loss(y, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_accuracy.update_state(y, pred)
            train_loss.update_state(loss)

        def test_step(x, y):
            pred = model(x, training=False)
            loss = compute_loss(y, pred)

            test_accuracy.update_state(y, pred)
            test_loss.update_state(loss)

        @tf.function
        def distributed_train_step(x, y):
            strategy.experimental_run_v2(train_step, args=(x, y,))

        @tf.function
        def distributed_test_step(x, y):
            strategy.experimental_run_v2(test_step, args=(x, y,))

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
                    distributed_test_step(x, y)

                with test_writer.as_default(): 
                    log_tensorboard("TEST", start, test_accuracy, 
                        epoch, test_loss, model, x)
                    reset_test_metrics() 

            # Train, but not the very last epoch as its no more used...
            if not is_last_epoch:
                for x, y in train_ds:
                    start = time.time()
                    distributed_train_step(x, y)

                with train_writer.as_default(): 
                    log_tensorboard("TRAIN", start, train_accuracy, epoch,
                        train_loss, model, x)
                    reset_train_metrics()
                train_writer.flush()        
        

def log_tensorboard(name, start, accuracy, epoch, loss, model, x):
        
    accuracy_val = accuracy.result().numpy()
    loss_val = loss.result().numpy()

    print("[%s] Epoch %d (%d): Loss %.7f ; Accuracy %.4f; Time/step %.3f" % (
        name, epoch, epoch,  loss_val,
        accuracy_val, time.time() - start), flush=True)

    tf.summary.scalar("Accuracy", accuracy_val, step=epoch)
    tf.summary.scalar("Loss", loss_val, step=epoch)

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

    if config.dtype != "float32":
        policy = mixed_precision.Policy('mixed_%s' % config.dtype)
        mixed_precision.set_policy(policy)

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