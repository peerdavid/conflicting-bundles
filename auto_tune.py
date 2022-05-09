#
# Automatically create the architecture and train the network. Can then be
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
import conflicting_bundle as cb
from config import get_config, save_config
config = get_config()

# Only one run as we want to see how auto-tune creates different architectures.
# Therefore auto_tune should be called multiple times with different log dirs
config.runs = 1

def prune_model(model, train_ds):
    """ If the bundle entropy is larger than zero, the first conflicting layer
        and all subsequent layers of the same block type (a, b, c or d) are
        removed from the architecture.
    """
    print("Start pruning of architecture...", flush=True)
    conflicts = cb.bundle_entropy(
            model, train_ds,
            config.batch_size, config.learning_rate,
            config.num_classes, config.conflicting_samples_size,
            all_layers=True)

    layer = -1
    pruned = False
    for block_type in range(len(config.pruned_layers)):
        new_layers = 0
        for block_layer in range(config.pruned_layers[block_type]):
            layer += 1
            bundle_entropy_of_layer = conflicts[layer][1]

            if bundle_entropy_of_layer <= 0:
                new_layers += 1
            else:
                pruned = True

        config.pruned_layers[block_type] = new_layers

        if pruned:
            save_config(config)
            return


def train(train_ds, test_ds, train_writer, test_writer, log_dir_run):

    # Initialize
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync

    with strategy.scope():
        model = create_model(config)
        radam=tfa.optimizers.RectifiedAdam(
            learning_rate=config.learning_rate,
            epsilon=1e-6, weight_decay=1e-2
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
        # Train and test step functions
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


    with strategy.scope():
        @tf.function
        def distributed_train_step(x, y):
            strategy.run(train_step, args=(x, y,))

        @tf.function
        def distributed_test_step(x, y):
            strategy.run(test_step, args=(x, y,))

        #
        # TRAINING LOOP
        #
        epoch = 0
        while True:
            if epoch > config.epochs:
                break

            is_last_epoch = epoch >= config.epochs-1
            print("", flush=True)
            model.save_weights("%s/ckpt-%d" % (log_dir_run, epoch))

            #
            # Test
            #
            if epoch % 5 == 0 or is_last_epoch:
                for x, y in test_ds:
                    start = time.time()
                    distributed_test_step(x, y)

                with test_writer.as_default():
                    log_tensorboard("TEST", start, test_accuracy,
                        epoch, test_loss, model, x)
                    reset_test_metrics()

            #
            # Train
            #
            if not is_last_epoch:
                for x, y in train_ds:
                    start = time.time()
                    distributed_train_step(x, y)

                with train_writer.as_default():
                    log_tensorboard("TRAIN", start, train_accuracy, epoch,
                        train_loss, model, x)
                    reset_train_metrics()
                train_writer.flush()

            # In the previous experiments we have seen that
            # conflicts occur after the third epoch. If we already
            # trained the network for 10 epochs without any conflicts,
            # we continue without checking conflicts because we have seen
            # in the experiments that after 10 epoch conflicts rarly occur.
            if epoch < 10:
                # We evaluate the conflicting layer only if we have
                # conflicts at a^{(L)} as its computationally cheaper to
                # evaluate only the last layer rather than all layers
                config.all_conflict_layers = False
                print("Checking conflicts...")
                conflicts = cb.bundle_entropy(
                    model, train_ds,
                    config.batch_size, config.learning_rate,
                    config.num_classes, config.conflicting_samples_size,
                    all_layers=False)
                print("Entropy: %.5f" % conflicts[-1][1])

                if conflicts[-1][1] > 0:
                    # Our model has conflicts, so lets autotune our model
                    # "The training is then restarted with the new pruned architecture."
                    print("Found conflicting layers.", flush=True)
                    prune_model(model, train_ds)
                    return False

            epoch += 1

        return True


def log_tensorboard(name, start, accuracy, epoch, loss, model, x):

    accuracy_val = accuracy.result().numpy()
    loss_val = loss.result().numpy()

    print("[%s] Epoch %d (%d): Loss %.7f ; Accuracy %.4f; Time/step %.3f" % (
        name, epoch, epoch,  loss_val,
        accuracy_val, time.time() - start), flush=True)

    tf.summary.scalar("Accuracy", accuracy_val, step=epoch)
    tf.summary.scalar("Loss", loss_val, step=epoch)

    # Log some images (use only gpu 0)
    x = x.values[0] if config.num_gpus > 1 else x
    tf.summary.image("Input data", x, step=epoch, max_outputs=3,)


#
# M A I N
#
def main():
    global config

    print("\n\n####################", flush=True)
    print("# AUTO TUNE %s" % config.log_dir, flush=True)
    print("####################\n", flush=True)

    train_csv = []
    test_csv = []

    # "First, the largest network from fig. 1 (120 layer) is trained without
    # residual connections."
    config.model = "vgg"
    config.pruned_layers = [3,12,41,3]
    config.use_residual = False
    config.conflicting_samples_size = 64
    log_dir_run = "%s/0" % (config.log_dir)

    # Log some things
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    train_writer = tf.summary.create_file_writer("%s/train" % log_dir_run)
    test_writer = tf.summary.create_file_writer("%s/test" % log_dir_run)

    # Load dataset
    train_ds, test_ds = load_dataset(config, augment=True)

    # "[...] This process is repeated until no conflicting layer can be found and the
    # network is successfully trained without conflicting bundles for 120 epochs."
    trained = False
    while not trained:
        print("###################################################")
        print("# Train model with blocks %s" % config.pruned_layers)
        print("###################################################")

        # Write hyperparameters, therefore in TensorBoard all
        # network updates are logged.
        with train_writer.as_default():
            params = vars(config)
            text = "|Parameter|Value|  \n"
            text += "|---------|-----|  \n"
            for val in params:
                text += "|%s|%s|  \n" % (val, params[val])
            tf.summary.text("Hyperparameters", text, step=0)
        train_writer.flush()

        # Try to train the model
        trained = train(train_ds, test_ds, train_writer, test_writer, log_dir_run)


if __name__ == '__main__':
    main()