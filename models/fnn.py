import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa



class FNN(tf.keras.Model):
    
    def __init__(self, config):
        super(FNN, self).__init__()

        self.fc_layers = []
        self.config = config
        self.image_dim = self.config.img_width * self.config.img_height * self.config.img_depth

        for l in range(config.num_layers):
            self.fc_layers.append(
                layers.Dense(config.width_layers, 
                name="fc%d" % l, 
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer=tf.zeros_initializer)
            )

        self.out = layers.Dense(config.num_classes, 
                name="out", 
                activation="linear")


    def call(self, x, training=True):
        outputs = []
        batch_size = tf.shape(x)[0]

        # Reshape input images
        x = tf.reshape(x, [batch_size, -1])

        # FC hidden layer
        for l in range(len(self.fc_layers)):
            x = self.fc_layers[l](x)
            outputs.append(x)

        # Output layer
        linear = self.out(x)
        outputs.append(linear)
        return outputs


    def max_weights(self):
        ret = []

        l=0
        for weights in self.fc_layers.trainable_weights:
            w = tf.reduce_max(tf.abs(weights))
            ret.append(w)
            l+=1

        return tf.reduce_max(ret)