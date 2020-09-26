import numpy as np
import tensorflow as tf


class VGGNet(tf.keras.Model):
    def __init__(self, layer_params, config, block_fn):
        super(VGGNet, self).__init__()
        self.config = config

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()


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

        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(
            units=config.num_classes, 
            activation="linear") # No softmax as this is part of the loss function!


    def call(self, inputs, training):
        outputs = []
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        for block in self.blocks:
            x = block(x, training=training)
            outputs.append(x)
        
        x = self.pool2(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        outputs.append(x)

        x = self.fc(x)
        outputs.append(x)

        return outputs
    

    def max_weights(self):
        ret = []

        for weights in self.conv1.trainable_weights:
            ret.append(tf.reduce_max(tf.abs(weights)))
        
        for block in self.blocks:
            for weights in block.trainable_weights:
                ret.append(tf.reduce_max(tf.abs(weights)))

        for weights in self.fc.trainable_weights:
                ret.append(tf.reduce_max(tf.abs(weights)))
        return tf.reduce_max(ret)