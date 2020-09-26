
import tensorflow as tf
from models.vggnet import VGGNet
from models.fnn import FNN
from models.vgg_block import make_basic_block_layer as make_vgg_block
from models.residual_block import make_basic_block_layer as make_residual_block


def create_model(config):
    # Create from fixed architecture
    if config.model == "vgg":
        blocks = get_blocks(config)
        return VGGNet(blocks, config, block_fn=make_vgg_block)
    elif config.model == "resnet":
        blocks = get_blocks(config)
        return VGGNet(blocks, config, block_fn=make_residual_block)
    elif config.model == "fnn":
        return FNN(config)

    raise Exception("Unknown model " + str(config.model))


def get_blocks(config):
    # If we get pruned layers from auto_tune, use those...
    if hasattr(config, "pruned_layers"):
        return config.pruned_layers

    if config.num_layers == 4:
        return [1, 0, 0, 0]
    elif config.num_layers == 10:
        return [1, 1, 1, 1]
    elif config.num_layers == 20:
        return [2, 2, 3, 2]
    elif config.num_layers == 30:
        return [3, 3, 5, 3]
    elif config.num_layers == 50:
        return [3, 6, 12, 3]
    elif config.num_layers == 76:
        return [3, 6, 25, 3]
    elif config.num_layers == 100:
        return [3, 12, 31, 3]
    elif config.num_layers == 120:
        return [3, 12, 41, 3]

    raise Exception("Unknown size " + str(config.num_layers))    