import tensorflow as tf


def soft_f1_binary(y_true, y_pred):
    """This is the standard version (F1 score only for the "1" class)"""
    tp = tf.math.reduce_sum(y_true * y_pred)
    #  tn = tf.math.reduce_sum((1.-y_true)*(1.-y_pred)) #  Note: Not used by soft_f1
    fp = tf.math.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.math.reduce_sum(y_true * (1.0 - y_pred))

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    return tf.math.reduce_mean(f1)


def soft_f1(y_true, y_pred):
    #   macro f1 score, i.e. average f1 score over all classes
    num_classes = y_pred.get_shape()[1]  #   use label shape

    soft_f1_multi_score = tf.zeros(1, dtype=tf.float32)
    for i in range(num_classes):
        soft_f1_multi_score += soft_f1_binary(y_true[:, i], y_pred[:, i])
    soft_f1_multi_score /= num_classes

    return soft_f1_multi_score


def soft_f1_loss(y_true, y_pred):
    return 1.0 - soft_f1(y_true, y_pred)


def soft_f1_binary_loss(y_true, y_pred):
    return 1.0 - soft_f1_binary(y_true, y_pred)
