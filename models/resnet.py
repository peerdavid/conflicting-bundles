# Credit: https://github.com/calmisential/TensorFlow2.0_ResNet

import numpy as np
import tensorflow as tf
from models.residual_block import make_basic_block_layer



class ResNet(tf.keras.Model):
    def __init__(self, layer_params, config):
        super(ResNet, self).__init__()
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
            self.pool1 = None
        
        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0],
                                             config=config)
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             config=config,
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             config=config,
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             config=config,
                                             stride=2)

        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(
            units=config.num_classes, 
            activation="linear") # No softmax as this is part of the loss function!


    def call(self, inputs, training):
        outputs = []
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        if self.pool1 != None:
            x = self.pool1(x)

        if self.layer1 != None:
            x, block_outputs = self._call_basic_block(self.layer1, x, training)
            outputs.extend(block_outputs)
        if self.layer2 != None:
            x, block_outputs = self._call_basic_block(self.layer2, x, training)
            outputs.extend(block_outputs)
        if self.layer3 != None:
            x, block_outputs = self._call_basic_block(self.layer3, x, training)
            outputs.extend(block_outputs)
        if self.layer4 != None:
            x, block_outputs = self._call_basic_block(self.layer4, x, training)
            outputs.extend(block_outputs)

        x = self.pool2(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        outputs.append(x)

        x = self.fc(x)
        outputs.append(x)

        return outputs
    

    def _call_basic_block(self, layer, x, training):
        # During training we make use of the sequential model...
        if training:
            return layer(x, training=training), [x]
        
        # ... but during evaluation we want to evaluate all a^{(l)}:
        block_outputs = []
        for basic_block in layer.layers:
            x = basic_block(x, training=training)
            block_outputs.append(x)
        return x, block_outputs
    

    def max_weights(self):
        ret = []

        for weights in self.conv1.trainable_weights:
            ret.append(tf.reduce_max(tf.abs(weights)))
        
        if self.layer1 != None:
            ret.append(self.max_block_weights(self.layer1.layers))
        if self.layer2 != None:
            ret.append(self.max_block_weights(self.layer2.layers))
        if self.layer3 != None:
            ret.append(self.max_block_weights(self.layer3.layers))
        if self.layer4 != None:
            ret.append(self.max_block_weights(self.layer4.layers))

        for weights in self.fc.trainable_weights:
                ret.append(tf.reduce_max(tf.abs(weights)))
        return tf.reduce_max(ret)


    def max_block_weights(self, layers):
        ret = []
        for layer in layers:
            for weights in layer.trainable_weights:
                ret.append(tf.reduce_max(tf.abs(weights)))
        return tf.reduce_max(ret)