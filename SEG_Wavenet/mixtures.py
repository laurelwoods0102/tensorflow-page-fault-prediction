import tensorflow as tf

@tf.function
def log_sum_exp(x):
    axis = len(x.shape) - 1
    m = tf.math.reduce_max(x, axis)
    m2 = tf.math.reduce_max(x, axis, keepdims=True)
    return m + tf.math.log(tf.math.reduce_sum(tf.math.exp(x - m2), axis))

@tf.function
def log_prob_from_logits(x):
    axis = len(x.shape) - 1
    m = tf.math.reduce_max(x, axis, keepdims=True)
    return x - m - tf.math.log(tf.math.reduce_sum(tf.math.exp(x - m), axis, keepdims=True))

@tf.function
def discretized_mix_logistic_loss(y_hat, y, num_class=2**8, log_scale_min=float(tf.math.log(1e-14)), reduce=True):
    y_hat_shape = y_hat.shape

    assert len(y_hat_shape) == 3
    assert y_hat_shape[2] % 3 == 0

    nr_mix = y_hat_shape[2] // 3

    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:nr_mix * 2]
    log_scales = tf.math.maximum(y_hat[:, :, nr_mix * 2:nr_mix * 3], log_scale_min)

    y = tf.tile(y, [1, 1, nr_mix])

    centered_y = y - means
    inv_std = tf.math.exp(-log_scales)

    plus_in = inv_std * (centered_y + 1. / (num_class - 1))
    cdf_plus = tf.nn.sigmoid(plus_in)
    minus_in = inv_std * (centered_y - 1. / (num_class - 1))
    cdf_minus = tf.nn.sigmoid(minus_in)

    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    log_one_minus_cdf_min = -tf.nn.softplus(minus_in)

    cdf_delta = cdf_plus - cdf_minus


    mid_in = inv_std * centered_y

    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    log_probs = tf.where(
        y < -0.999, 
        log_cdf_plus,
        tf.where(
            y > 0.999, 
            log_one_minus_cdf_min,
            tf.where(
                cdf_delta > 1e-5,
                tf.math.log(tf.math.maximum(cdf_delta, 1e-12)),
                log_pdf_mid - tf.math.log((num_class - 1) / 2)
                )
            )
        )
    log_probs = log_probs + tf.nn.log_softmax(logit_probs, -1)

    if reduce:
        return -tf.math.reduce_sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs)

@tf.function
def sample_from_discretized_mix_logistic(y, log_scale_min=float(tf.math.log(1e-14))):
    y_shape = y.shape

    assert len(y_shape) == 3
    assert y_shape[2] % 3 == 0

    nr_mix = y_shape[2] // 3

    logit_probs = y[:, :, :nr_mix]

    sel = tf.one_hot(tf.argmax(logit_probs - tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logit_probs), minval=1e-5, maxval=1. - 1e-5))), 2), depth=nr_mix, dtype=tf.float32)

    means = tf.math.reduce_sum(y[:, :, nr_mix:nr_mix * 2] * sel, axis=2)

    log_scales = tf.math.maximum(tf.math.reduce_sum(y[:, :, nr_mix * 2:nr_mix * 3] * sel, axis=2), log_scale_min)

    u = tf.random.uniform(tf.shape(means), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.math.exp(log_scales) * (tf.math.log(u) - tf.math.log(1. - u))

    x = tf.math.minimum(tf.math.maximum(x, -1.), 1.)
    return x