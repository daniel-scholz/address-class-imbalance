import tensorflow as tf


def soft_mcc_binary(y_true, y_pred):

    tp = tf.cast(tf.reduce_sum(y_true * y_pred), dtype=tf.float32)
    tn = tf.cast(
        tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred)), tf.float32
    )  #  Note: Not used by soft_f1
    fp = tf.cast(tf.reduce_sum((1.0 - y_true) * y_pred), dtype=tf.float32)
    fn = tf.cast(tf.reduce_sum(y_true * (1.0 - y_pred)), dtype=tf.float32)

    #   tf.print(tp, tn, fp, fn)
    numerator = (tp * tn) - (fp * fn)
    denom = (
        tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        + tf.keras.backend.epsilon()
    )
    soft_mcc = numerator / denom

    return tf.reduce_mean(soft_mcc)


def soft_mcc_binary_loss(y_true, y_pred):
    return 1.0 - soft_mcc_binary(y_true, y_pred)


def soft_mcc(y_true, y_pred):

    c = tf.reduce_sum(y_true * y_pred)

    s = tf.shape(y_pred)[0]
    s = tf.cast(s, dtype=tf.float32)

    t_k = tf.reduce_sum(y_true, axis=0)
    p_k = tf.reduce_sum(y_pred, axis=0)

    numerator = c * s - tf.reduce_sum(t_k * p_k)
    s_squared = s**2

    denom = (
        tf.sqrt(s_squared - tf.reduce_sum(p_k**2))
        * tf.sqrt(s_squared - tf.reduce_sum(t_k**2))
        + tf.keras.backend.epsilon()
    )

    soft_mcc = numerator / denom

    return soft_mcc


def soft_mcc_loss(y_true, y_pred):
    return 1.0 - soft_mcc(y_true, y_pred)
