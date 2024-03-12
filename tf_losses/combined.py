import tensorflow as tf

from tf_losses.soft_f1 import soft_f1_binary_loss, soft_f1_loss
from tf_losses.soft_mcc import soft_mcc_binary_loss, soft_mcc_loss


def ce_mcc_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + soft_mcc_loss(
        y_true, y_pred
    )


def ce_f1_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + soft_f1_loss(
        y_true, y_pred
    )


def bce_mcc_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + soft_mcc_binary_loss(
        y_true, y_pred
    )


def bce_f1_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + soft_f1_binary_loss(
        y_true, y_pred
    )
