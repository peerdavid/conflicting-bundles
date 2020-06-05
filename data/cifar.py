
import tensorflow as tf
import tensorflow_datasets as tfds



class Cifar:

    def __init__(self, config, augment):
        config.num_classes = 10
        config.img_depth = 3
        config.img_width = 32
        config.img_height = 32
        self.augment_fn = self._augment if augment else self._no_augment
        self.config = config


    def load_dataset(self):        
        train_ds = tfds.load(name="cifar10", split="train", as_supervised=True)

        test_ds = tfds.load(name="cifar10", split="test", as_supervised=True)
        train_ds = (train_ds
            .shuffle(buffer_size=10000, reshuffle_each_iteration=True)
            .map(self.augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.config.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
        test_ds = (test_ds
            .map(self._no_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.config.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
        return train_ds, test_ds
    

    def _augment(self, images, labels):
        images = tf.image.convert_image_dtype(images, tf.float32)
        images = tf.image.per_image_standardization(images)
        images = tf.image.resize_with_pad(images, target_height=self.config.img_height+6, target_width=self.config.img_width+6)
        images = tf.image.random_crop(images, size=[self.config.img_height, self.config.img_width, self.config.img_depth])
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, max_delta=0.25) # Random brightness
        return images, labels


    def _no_augment(self, images, labels):
        images = tf.image.convert_image_dtype(images, tf.float32)
        images = tf.image.per_image_standardization(images)
        return images, labels