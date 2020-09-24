

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def bundle_entropy(train_ds, model, config):
    """ Given a dataset train_ds and a model, this function returns
        foreach a^l the number of bundles and the bundle entropy at time 
        step t. 
        
        param: train_ds - Training dataset
        param: model - The neural network which returns a list [a^1, a^2, ... a^L, logits]
        param: config.all_conflict_layers - False to evaluate only a^L, otherwise a^1 to a^L are evaluated
        param: config.conflicting_samples_size - Subset size of train_ds to use for bundle calculation
        param: config.num_classes - Number of classes needed to calculate the bundle entropy
        
        returns: [[num_bundle_a^1, bundle_entropy_a^1], ... [num_bundle_a^L, bundle_entropy_a^L]]
                 Please note that the logits are not part of the return value as this is 
                 also not defined in the paper.
    """
    layer_eval = 0 if config.all_conflict_layers else -2
    A, Y = [], []

    @tf.function
    def inference(x):
        return model(x, training=False)
    
    for x, y in train_ds:
        if len(Y) * config.batch_size >= config.conflicting_samples_size:
                continue

        layers = inference(x)
        
        # Input is of type [a^(1), a^(2), ...  a^{(L)}, logits]
        # Following the definition of conflicts and bundles from the paper 
        # we therefore access the array from n to :-1 and exclude the logits:
        # "The last softmax layer is per definition not included."
        A.append(layers[layer_eval:-1])
        if config.num_gpus > 1:
            Y.append(tf.concat(y, axis=0))
        else:
            Y.append(y)
    
    Y = tf.concat(Y, axis=0)

    # If needed we return the conflicts for each layer. For evaluation only 
    # a^{(L)} is needed, but e.g. to use it with auto-tune all conflicting 
    # layers are needed. Therefore if all layers are evaluated the complexity
    # is O(L * |X|)
    res = []
    max_weights = model.max_weights()
    A = zip(*A)
    iterator = tqdm(A) if config.all_conflict_layers else A
    for a in iterator:
        a = tf.concat(a, axis=0)

        # As written in the paper we not directly compare a_i and a_j in order 
        # to consider also floating point representations during backpropagation
        # Instead of doing an inequallity check using \gamma, we do an equality
        # check after subtracting the values from the maximum weights which 
        # is equivalent. Note that this is not possible if gamma should be 
        # larger than the floating point resolution.
        equality_check = max_weights - config.learning_rate * a * 1 / config.batch_size
        equality_check = tf.reshape(equality_check, [tf.shape(equality_check)[0], -1])
        num_bundle, bundle_entropy = get_bundle_and_conflicts(equality_check, Y, config)
        res.append([num_bundle, bundle_entropy])

    return res


def get_bundle_and_conflicts(X, Y, config):
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

        bundle, bundle_entropy = calculate_bundle(
            i, bundle, X, Y, bundle_entropy, config)

    num_bundles = tf.reduce_max(bundle)
    return num_bundles, bundle_entropy / float(tf.shape(X)[0])
    

def calculate_bundle(i, bundle, X, Y, bundle_entropy, config):
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

    # Calculate the bundle entropy for the current bundle (same_bundle) using 
    # the entropy
    bundle_only = tf.cast(tf.boolean_mask(Y, same_bundle), tf.int32)
    bundle_class_prob = tf.math.bincount(bundle_only, minlength=config.num_classes, maxlength=config.num_classes, dtype=tf.float32)
    bundle_class_prob /= tf.reduce_sum(bundle_class_prob)
    bundle_size = tf.cast(tf.reduce_sum(same_bundle), tf.float32)
    entropy = -tf.reduce_sum(bundle_class_prob * tf.math.log(bundle_class_prob+1e-5), axis=-1)
    bundle_entropy += tf.maximum(0.0, entropy) * bundle_size

    return bundle, bundle_entropy
