import tensorflow as tf
import random
import numpy as np

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, config, stride=1):
        super(BasicBlock, self).__init__()
        self.config = config

        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            kernel_initializer="he_normal",
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            kernel_initializer="he_normal",
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(1, 1),
                                                    kernel_initializer="he_normal",
                                                    strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x
            # Here also other bijective functions can be used. We tested with
            # self.downsample = lambda x: 2*x+0.1  
    
    def call(self, inputs, training, use_residual=True):
        x = self.conv1(inputs)
        scale_bn1 = tf.reduce_mean(x) / tf.math.sqrt(tf.math.reduce_variance(x)+1e-6)

        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        
        scale_bn2 = 1.0 / tf.math.sqrt(tf.math.reduce_variance(x)+1e-6)
        self.scale = scale_bn2 * scale_bn1

        x = self.bn2(x, training=training)
        output_without_residual = tf.nn.relu(x)
        
        residual = self.downsample(inputs)
        if use_residual or training:
            x = tf.keras.layers.add([residual, x])
        else:
            # Kick out layer for lesion experiment
            x = residual

        output = tf.nn.relu(x)

        self.cb = [output, output_without_residual]
        return output

def make_basic_block_layer(filter_num, blocks, config, stride=1):
    ret = []
    if blocks <= 0:
        return ret

    # "The first block of each new block type uses a stride of two"
    ret.append(BasicBlock(filter_num, config, stride=stride))
    for _ in range(1, blocks):
        ret.append(BasicBlock(filter_num, config, stride=1))

    return ret
