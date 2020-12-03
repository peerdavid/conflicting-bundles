"""
Implementation of conflicting bundles. See also https://arxiv.org/abs/2011.02956

Usage:
import conflicting_bundles as cb
[...]
cb = cb.bundle_entropy(model, ds, train_batch_size=64, train_learning_rate=1e-3, num_classes=10)
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math


def bundle_entropy(model, train_ds, train_batch_size, train_lr, 
                   num_classes, evaluation_size=512, all_layers=False):
    """ Given a dataset train_ds and a model, this function returns
        foreach a^l the number of bundles and the bundle entropy at time 
        step t. 
        
        Limitation: This implementation is currently only for a single
        GPU. I.e. you can train your model with multiple GPUs, and evaluate 
        cb with a single gpu.
        
        param: model - The model that should be evaluated. Note: We assume that model.cb exists.
                       model.cb is a list of tuples with (a, layer) pairs.
                       E.g.: model.cb = [(a_1, hidden_layer_1), (a_2, hidden_layer_2)]                        
        param: train_ds - Training dataset. Its important to NOT use the test set as we want to check 
                          how the training was negatively influenced. See https://arxiv.org/abs/2011.02956
        param: num_classes - Number of classes needed to calculate the bundle entropy
        param: train_batch_size - The batch size that was used for training. See https://arxiv.org/abs/2011.02956
        param: train_lr - The batch size that was used for training. See https://arxiv.org/abs/2011.02956
        param: evaluation_size - Subset size of train_ds to use for bundle calculation
        param: all_layers - False to evaluate only a^L, otherwise a^1 to a^L are evaluated
        
        returns: [[num_bundles_1, bundle_entropy_1], ... [num_bundles_L, bundle_entropy_L]]
    """
    train_batch_size = float(train_batch_size)
    layer_eval = 0 if all_layers else -1
    A, Y = [], []

    # Execute in eager mode to get access to model.cb
    def inference(x):
        model(x, training=False)

    for x, y in train_ds:
        if len(Y) * train_batch_size >= evaluation_size:
                continue
        
        inference(x)

        if not hasattr(model, 'cb'):
            print("(Warning) The provided model has no cb attribute set.")
            print("Please set a^(l) values in the array cb to measure the bundle entropy.")
            return None
            
        cb = model.cb[layer_eval:]
        layers = [c[1] for c in cb]
        A.append([c[0] for c in cb])
        Y.append(y)
    
    Y = tf.concat(Y, axis=0)

    # If needed we return the conflicts for each layer. For evaluation only 
    # a^{(L)} is needed, but e.g. to use it with auto-tune all conflicting 
    # layers are needed. Therefore if all layers are evaluated the complexity
    # is O(L * |X|)
    res = []
    A = zip(*A)
    for i, a in enumerate(A):
        a = tf.cast(tf.concat(a, axis=0), tf.float32)
        a = tf.concat(a, axis=0)

        # As written in the paper we not directly compare a_i and a_j in order 
        # to consider also floating point representations during backpropagation
        # Instead of doing an inequallity check using \gamma, we do an equality
        # check after subtracting the values from the maximum weights which 
        # is equivalent. Note that this is not possible if gamma should be 
        # larger than the floating point resolution.
        weights_amplitude = _get_weight_amplitude(layers[i])

        equality_check = weights_amplitude - a * train_lr * (1.0 / train_batch_size)
        equality_check = tf.reshape(equality_check, [tf.shape(equality_check)[0], -1])
        num_bundle, bundle_entropy = _calculate_bundles(equality_check, Y, num_classes)
        res.append([num_bundle, bundle_entropy])
        print("%d, %d, %d, %.5f" % (int(i/2), i%2, num_bundle, bundle_entropy), flush=True)

    return res


def _get_weight_amplitude(layer):
    """ With this function we approximate the weight
        amplitude of a layer. A layer could consist of multiple 
        sub layers (e.g. a vgg block with multiple layers).
        Therefore, we take the mean amplitude of each weight in 
        trainable_weights.

        param: layer - Layer for which the amplitude should be known
        return: Single floating point value of the max. weight amplitude of some sub layers
    """
    if not hasattr(layer, 'trainable_weights') or len(layer.trainable_weights) <= 0:
        return 0.0

    ret = []
    for weights in layer.trainable_weights:
        ret.append(tf.reduce_max(tf.abs(weights)))
    return tf.reduce_max(ret)


def _calculate_bundles(X, Y, num_classes):
    """ Calculates all bundles of X and calculates the bundle entropy using 
        the label information from Y. Iterates over X and therefore the 
        complexity is O(X) if calculate_bundle can be fully vectorized.
    """    

    # Create array for each feature and assign a unique bundlenumber...
    bundle = tf.zeros(tf.shape(X)[0], dtype=tf.float32)
    bundle_entropy = tf.constant(0.0, dtype=tf.float32)
    for i in range(len(X)):
        if bundle[i] != 0:
            continue

        bundle, bundle_entropy = _calculate_single_bundle(
            i, bundle, X, Y, bundle_entropy, num_classes)

    num_bundles = tf.reduce_max(bundle)
    return num_bundles, bundle_entropy / float(tf.shape(X)[0])
    

def _calculate_single_bundle(i, bundle, X, Y, bundle_entropy, num_classes):
    """ This function calculates a bundle which contains all x which are similar
        than X[i] using vectorization such that only O(|X|) is needed to calculate
        bundles for one layer. The idea is to use equality operators, mask 
        all others out, set the bundle and reuse this information vector 
        for the next samples. As a vector contains all equivalent samples
        also the bundle entropy at time step t can immediately be calculated.
    """
    dim_X = tf.cast(tf.shape(X)[1], tf.float32)
    next_bundle_id = tf.reduce_max(bundle) + tf.constant(1.0, dtype=tf.float32)
    num_samples = tf.shape(X)[0]
    x = X[i]

    # Ignore all that are already bundleed (bundle > 0)
    zero_out = tf.cast(tf.less_equal(bundle, 0.0), tf.float32)                             
    # Set bundle id for current x[i]
    bundle += tf.one_hot(i, num_samples, dtype=tf.float32) * next_bundle_id * zero_out   
     # Also ignore current x[i] 
    zero_out = tf.cast(tf.less_equal(bundle, 0.0), tf.float32)                            
    # Get all equivalent components (single component only of possible ndim inputs)
    equal_components = tf.cast(tf.equal(x, X), tf.float32)            
    # All components must be equivalent, therefore check (using the sum and dim_X) whether this is the case                      
    num_equal_components = tf.reduce_sum(equal_components, axis=-1)                         
    same_bundle = tf.cast(tf.greater_equal(num_equal_components, dim_X), tf.float32)
    # And bundle all components
    bundle += same_bundle * next_bundle_id * zero_out

    # Calculate the bundle entropy for the current bundle (same_bundle) using the entropy
    bundle_only = tf.cast(tf.boolean_mask(Y, same_bundle), tf.int32)
    bundle_class_prob = tf.math.bincount(bundle_only, minlength=num_classes, maxlength=num_classes, dtype=tf.float32)
    bundle_class_prob /= tf.reduce_sum(bundle_class_prob)
    bundle_size = tf.cast(tf.reduce_sum(same_bundle), tf.float32)
    entropy = -tf.reduce_sum(bundle_class_prob * tf.math.log(bundle_class_prob+1e-5), axis=-1)
    bundle_entropy += tf.maximum(0.0, entropy) * bundle_size

    return bundle, bundle_entropy
