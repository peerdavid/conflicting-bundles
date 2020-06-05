
import os
import argparse
import tensorflow as tf
import numpy as np
import pickle


#
# Hyperparameters and cmd args
#
def get_config():
    argparser = argparse.ArgumentParser(description="Conflicting Clusters")

    # Training
    argparser.add_argument("--log_dir", default="tmp",  help="Log directory")   
    argparser.add_argument("--dataset", default="imagenette", help="imagenette, cifar, svhn or mnist")
    argparser.add_argument("--runs", default=3, type=int, help="Multiple executions to get mean and std. Ignored for auto-tune in order to get multiple architectures from different executions")
    argparser.add_argument("--epochs", default=120, type=int, help="Number of epochs")
    argparser.add_argument("--batch_size", default=64, type=int, help="Batch size used for training")
    argparser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate for optimizer")

    # Model
    argparser.add_argument("--num_layers", default=50, type=int, help="Number of layers for ResNet. Ignored for auto-tune training.")
    argparser.add_argument("--use_residual", default="False", help="Should a residual connection be used? Ignored for auto-tune training.")

    # Evaluation
    argparser.add_argument("--last_epoch_only", default="False", help="Evaluate only last epoch.")
    argparser.add_argument("--conflicting_samples_size", default=2048, help="How many samples are used for conflict test. Ignored for auto-tune training.")
    argparser.add_argument("--all_conflict_layers", default="False", help="Evaluate conflicts of each layer. Ignored for auto-tune training.")

    # Update params
    config = argparser.parse_args()

    # Load from file or from args
    log_dir = "experiments/%s/%s" % (config.dataset, config.log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Load config from previous run or parse cmd args
    config.file = "%s/config.pkl" % log_dir
    file_config = _load_config_from_file(config)
    if file_config != None:
        file_config.all_conflict_layers = _str_to_bool(config.all_conflict_layers)
        return file_config

    config.num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    config.use_residual = _str_to_bool(config.use_residual)
    config.last_epoch_only = _str_to_bool(config.last_epoch_only)
    config.all_conflict_layers = _str_to_bool(config.all_conflict_layers)
    config.conflicting_samples_size = int(config.conflicting_samples_size)
    config.log_dir = log_dir
    save_config(config)

    return config


def save_config(config):
    with open(config.file, "wb") as conf_file: 
        pickle.dump(config, conf_file, pickle.HIGHEST_PROTOCOL)


def _load_config_from_file(config):
    if os.path.exists(config.file):
        with open(config.file, "rb") as conf_file:
            config = pickle.load(conf_file)
            return config
    return None


def _str_to_bool(value):
    return True if value.lower() == "true" else False