import numpy as np
import tensorflow as tf
import random

class VGGNet(tf.keras.Model):
    def __init__(self, layer_params, config, block_fn, lesion=None):
        super(VGGNet, self).__init__()
        self.config = config
        self.scale=1.0
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.use_residual = []

        # "The MaxPool2D 3x3 operator is not added
        # for small datasets such as cifar where each image has only 32 × 32px"
        if config.img_width > 50:
            self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                    strides=2,
                                                    padding="same")
        else:
            self.pool1 = lambda x: x
        
        self.blocks = []
        self.blocks.extend(block_fn(filter_num=64,
                                blocks=layer_params[0],
                                config=config))
        self.blocks.extend(block_fn(filter_num=128,
                                blocks=layer_params[1],
                                config=config,
                                stride=2))
        self.blocks.extend(block_fn(filter_num=256,
                                blocks=layer_params[2],
                                config=config,
                                stride=2))
        self.blocks.extend(block_fn(filter_num=512,
                                blocks=layer_params[3],
                                config=config,
                                stride=2))
        
        #nop = random.randint(0, len(self.blocks))
        #lesion_pos = random.sample(range(0, len(self.blocks)), lesion)
        if lesion is not None:
            for i in range(len(self.blocks)):
                self.use_residual.append(not i in lesion)
        
        #print(self.use_residual, flush=True)

        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(
            units=config.num_classes, 
            activation="linear") # No softmax as this is part of the loss function!


    def call(self, inputs, training):
        self.cb = []
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        for i, block in enumerate(self.blocks):
            use_residual = True if len(self.use_residual) == 0 else self.use_residual[i]
            x = block(x, training=training, use_residual=use_residual)
            
            for c in block.cb:
                self.cb.append((c, block))
        
        x = self.pool2(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        self.cb.append((x, self.pool2))

        x = self.fc(x)
        return x