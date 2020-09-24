
import tensorflow as tf
from models.resnet import ResNet
from models.fnn import FNN


def create_model(config):
    # Create model from auto-tune
    if hasattr(config, "pruned_layers"):
        return ResNet(config.pruned_layers, config)
    
    # Create from fixed architecture
    if config.model == "vgg" or config.model == "resnet":
        return create_resnet_vggnet(config)
    elif config.model == "fnn":
        return FNN(config)

    raise Exception("Unknown model " + str(config.model))


def create_resnet_vggnet(config):
    if config.num_layers == 4:
        return ResNet([1, 0, 0, 0], config)
    elif config.num_layers == 10:
        return ResNet([1, 1, 1, 1], config)
    elif config.num_layers == 20:
        return ResNet([2, 2, 3, 2], config)
    elif config.num_layers == 30:
        return ResNet([3, 3, 5, 3], config)
    elif config.num_layers == 50:
        return ResNet([3, 6, 12, 3], config)
    elif config.num_layers == 76:
        return ResNet([3, 6, 25, 3], config)
    elif config.num_layers == 100:
        return ResNet([3, 12, 31, 3], config)
    elif config.num_layers == 120:
        return ResNet([3, 12, 41, 3], config)
    
    raise Exception("Unknown resnet " + str(config.num_layers))