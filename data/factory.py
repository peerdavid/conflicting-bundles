
from data.mnist import MNIST
from data.imagenette import Imagenette
from data.cifar import Cifar
from data.svhn import Svhn

def load_dataset(config, augment):
    if config.dataset == "mnist":
        data_provider = MNIST(config, augment)
    elif config.dataset == "imagenette":
        data_provider = Imagenette(config, augment)
    elif config.dataset == "svhn":
        data_provider = Svhn(config, augment)
    elif config.dataset == "cifar":
        data_provider = Cifar(config, augment)
        
    else:
        raise Exception("Unknown dataset %s" % config.dataset)

    return data_provider.load_dataset()