from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.keras import losses
import tensorflow as tf
#from tensorflow.keras.losses import mean_squared_error
import numpy as np

# for debugging change to True
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

#######################################################################################
# Note that Keras Backend functions and Tensorflow mathematical operations will be used 
# instead of numpy functions to avoid some silly errors. 
# Keras backend functions work similarly to numpy functions.
# see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_ops.py
#######################################################################################


def monotonicity_loss1(y_pred):
    # penalizes if the weighted average of predictions f(x_i)​ decreases over time
    ### R(f(x)) = 1/N-1 * ∑i (max{0, <f(x_i)> - <f(x_i+1)>})
    weighted_avg_pred = tf.reduce_sum([y_pred[:,i] * i for i in range(y_pred.shape[1])], axis=0)
    diff = tf.maximum(0.0, weighted_avg_pred[:-1] - weighted_avg_pred[1:])
    return tf.reduce_mean(diff)

def monotonicity_loss2(y_pred):
    # penalizes large differences between consecutive predictions for smooth transitions
    # R(f(x)) = 1/N-1 * ∑i (<f(x_i)> - <f(x_i+1)>)^2
    weighted_avg_pred = tf.reduce_sum([y_pred[:,i] * i for i in range(y_pred.shape[1])], axis=0)
    diff = K.square(weighted_avg_pred[:-1] - weighted_avg_pred[1:])
    return tf.reduce_mean(diff)

def monotonicity_loss3(y_pred):
    # ensures that the cumulative mass at each index does not decrease over time
    # R(f(x)) = 1/N-1 * ∑i ∑k (max{0, C(f(x_i)[k]) - C(f(x_i+1)[k])})
    # C(f(x_i)[k]) = ∑j=1 to k (f(x_i)[j])
    cumm = tf.reduce_sum([y_pred[:,:k] for k in range(y_pred.shape[1])], axis=0)
    diff = tf.maximum(0.0, cumm[:-1] - cumm[1:])
    return tf.reduce_mean(diff)

def monotonicity_loss4(y_pred):
    # account for the shift in the output distribution without calculating the weighted average
    # R(f(x)) = 1/N-1 * ∑i (max{0, ∑k k * (f(x_i)[k] - f(x_i+1)[k])})
    kl = tf.reduce_sum([y_pred[:-1,k]*K.log(y_pred[:-1,k]/y_pred[1:,k]) for k in range(y_pred.shape[1])], axis=0)
    return tf.reduce_mean(kl)


#def informed_loss(i, information, loss_fun, sample):
def informed_loss(information, loss_fun, sample):

    if loss_fun=='mse':
        loss_fun = tf.keras.losses.MeanSquaredError()
    elif loss_fun=='monotonic_mse':
        loss_fun = monotonic_mse
    elif loss_fun=='cce':
        loss_fun = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    elif loss_fun=='monotonic_cce':
        print("loss:", {loss_fun})
        loss_fun = monotonic_categorical_crossentropy
    elif loss_fun=='kld':
        loss_fun = tf.keras.losses.KLDivergence()
    elif loss_fun=='monotonic_kld':
        loss_fun = monotonic_kl_divergence

    def loss(data, y_pred):
        print(f"Data structure: {type(data)}, Length: {len(data)}")
        # batch_inputs, y_true = data # extract images and labels from combined data
        # y_true = tf.squeeze(y_true, axis=[1, 3])  # Reshape to [batch_size, 380]

        batch_inputs = data[:, :, :220, :]#data[:, 1]
        y_true = data[:, :, 220:, :]#data[:, 0]
        y_true = y_true[:, 0, :, 0]
        y_pred = y_pred[:, :, 220:, :][:, 0, :, 0]

        # pred_weighted_avg  = tf.reduce_sum([y_pred[:,i] * sample[i] for i in range(y_pred.shape[1])], axis=0)
        # penalty = information(batch_inputs, pred_weighted_avg)
        # return loss_fun(y_true, y_pred) + 0.00001*tf.reduce_mean(penalty)
        pred_weighted_avg = tf.reduce_sum(
            [y_pred[:, i] * sample[i] for i in range(y_pred.shape[1])], axis=0
        )
        # calculate penalty using information function

        penalty = information(batch_inputs, pred_weighted_avg)

        # Calculate loss
        return loss_fun(y_true, y_pred) + 0.00001*tf.reduce_mean(penalty)

   
    return loss

def monotonic_mse(y_true, y_pred):
    # standard MSE loss
    mse_loss = tf.keras.losses.MeanSquaredError()
    mse = mse_loss(y_true, y_pred)
    penalty = monotonicity_loss1(y_pred)
    
    # return the combined loss: MSE + monotonicity penalty
    return mse + 0.00001*penalty#0.00001#10#100#10#0.1

def monotonic_categorical_crossentropy(y_true, y_pred):
    cross_entropy = losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)
    # diff = y_pred[:, 1:] - y_pred[:, :-1]
    # factor = tf.range(0, tf.shape(y_pred)[1], dtype=tf.float32) / 100
    # y_pred_means = K.mean(y_pred * factor,  axis=-1)
    # y_true_means = K.mean(y_true * factor, axis=-1)
    # diff = y_pred_means - y_true_means
    # diff_2 = y_pred_means[1:] - y_pred_means[:-1]
    # diff_3 = y_true_means[1:] - y_pred_means[:-1]

    penalty = monotonicity_loss1(y_pred)

    # penalty = K.sum(K.maximum(-diff, 0))  # penalize decreases
    # penalty_2 = K.sum(K.maximum(-diff_2, 0)) 
    # penalty_3 = K.sum(K.maximum(-diff_3, 0)) 

    return cross_entropy + 1*penalty#K.sqrt(penalty)

def monotonic_kl_divergence(y_true, y_pred):
    # calculate KL Divergence
    # kl_divergence = losses.kl_divergence(y_true, y_pred)
    # kl_divergence = K.sum(y_true * K.log(y_true / (y_pred + K.epsilon())))
    
    # make sure the array does not exceeds or is less than the values
    y_t = K.clip(y_true, K.epsilon(), 1)
    y_p = K.clip(y_pred, K.epsilon(), 1)
    kl_divergence = K.sum(y_t * K.log(y_t / y_p), axis=-1)
    penalty = monotonicity_loss1(y_pred)

    # calculate the difference between predicted probabilities for consecutive classes
    # diff = y_pred[:, 1:] - y_pred[:, :-1]
    # factor = tf.range(0, tf.shape(y_pred)[1], dtype=tf.float32) / 100
    # y_pred_means = K.mean(y_pred * factor,  axis=-1)
    # y_true_means = K.mean(y_true * factor, axis=-1)
    # diff = y_pred_means - y_true_means
    # diff_2 = y_pred_means[1:] - y_pred_means[:-1]
    # diff_3 = y_true_means[1:] - y_pred_means[:-1]

    # apply penalty for non-monotonicity (i.e., when the difference is negative)
    # penalty = K.sum(K.maximum(-diff, 0))  # penalize decreases in predicted probabilities
    # penalty_2 = K.sum(K.maximum(-diff_2, 0)) 
    # penalty_3 = K.sum(K.maximum(-diff_3, 0)) 

    # return the combined loss: KL Divergence + monotonicity penalty
    return K.mean(kl_divergence) + 0.00001*penalty