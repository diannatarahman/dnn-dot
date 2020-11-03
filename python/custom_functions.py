from keras import backend as K, activations, losses

lrelu_alpha = 0.2

def custom_mean_squared_error(y_true, y_pred):
    nonzero = K.cast(K.not_equal(y_true, 0), K.floatx())
    return K.sum(K.square(y_pred * nonzero - y_true), axis=-1) / K.sum(nonzero, axis=-1)


def custom_mse_cosine(y_true, y_pred):
    loss1 = losses.mean_squared_error(y_true, y_pred)
    loss2 = losses.cosine_proximity(y_true, y_pred)
    return loss1 * (1 + 2 * (1 - K.exp(-5 * (loss2 + 2))))


def custom_binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)


def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true * y_pred, axis=-1)


def wasserstein_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true >= 0, y_pred >= 0), axis=-1)


def leaky_relu(x):
    return activations.relu(x, alpha=lrelu_alpha)


def l2_error_norm(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def multiply_constant(x, constant, adder=0):
    return constant*(x+adder)


def multiply_constant_reciprocal(x, constant, adder=0):
    return constant/(x+adder)


cmse = custom_mean_squared_error
cmc = custom_mse_cosine
cacc = custom_binary_accuracy
