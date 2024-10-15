from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.keras import losses

#######################################################################################
# Note that Keras Backend functions and Tensorflow mathematical operations will be used 
# instead of numpy functions to avoid some silly errors. 
# Keras backend functions work similarly to numpy functions.
# see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_ops.py
#######################################################################################

def monotonic_mse(y_true, y_pred):
    # standard MSE loss
    mse = K.mean(K.square(y_true - y_pred))
    # calculate the difference between consecutive predictions
    diff = y_pred[:, 1:] - y_pred[:, :-1]
    # apply penalty for non-monotonicity (i.e., when the difference is negative)
    penalty = K.sum(K.maximum(-diff, 0))  # penalize decreases in predictions
    # return the combined loss: MSE + monotonicity penalty
    return mse + penalty

def monotonic_categorical_crossentropy(y_true, y_pred):
    cross_entropy = losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)
    diff = y_pred[:, 1:] - y_pred[:, :-1]
    penalty = K.sum(K.maximum(-diff, 0))  # penalize decreases
    return cross_entropy + penalty

def monotonic_kl_divergence(y_true, y_pred):
    # calculate KL Divergence
    kl_divergence = K.sum(y_true * K.log(y_true / (y_pred + K.epsilon())))
    # calculate the difference between predicted probabilities for consecutive classes
    diff = y_pred[:, 1:] - y_pred[:, :-1]
    # apply penalty for non-monotonicity (i.e., when the difference is negative)
    penalty = K.sum(K.maximum(-diff, 0))  # penalize decreases in predicted probabilities
    # return the combined loss: KL Divergence + monotonicity penalty
    return K.mean(kl_divergence + penalty)