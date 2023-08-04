import tensorflow as tf
from tensorflow.keras import layers as KL

#TODO: make code consistent to other models
class Encoder(tf.keras.layers.Layer):

    def __init__(self,

                 dims=[63,500,500,2000,10],

                 act="relu"):

        super(Encoder, self).__init__()

        self.dims = dims

        self.act = act

    def call(self, inputs):

        n_stacks = len(self.dims) - 1

        h = inputs

        # internal layers in encoder
        for i in range(n_stacks - 1):
            h = KL.Dense(self.dims[i + 1], activation=self.act, name='encoder_%d' % i)(h)

        # hidden layer
        h = KL.Dense(self.dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

        return h

class Decoder(tf.keras.layers.Layer):
    def __init__(self,

                 dims=[63, 500, 500, 2000, 10],

                 act="relu"):
        super(Decoder, self).__init__()

        self.dims = dims

        self.act = act

    def call(self, inputs):

        n_stacks = len(self.dims) - 1

        h = inputs

        for i in range(n_stacks - 1, 0, -1):
            h = KL.Dense(self.dims[i], activation=self.act, name='decoder_%d' % i)(h)

        # output
        h = KL.Dense(self.dims[0], name='decoder_0')(h)

        return h



class DenseAutoencoder(tf.keras.Model):

    def __init__(self,

                 dims=[63, 500, 500, 2000, 10],

                 act="relu"):
        super(DenseAutoencoder, self).__init__()

        self.dims = dims

        self.act = act

        self.Encoder = Encoder(dims=self.dims, act=self.act)

        self.Decoder = Decoder(dims=self.dims, act=self.act)

    def call(self, inputs):

        logits = self.Encoder(inputs)

        output = self.Decoder(logits)

        return logits, output


def autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = tf.keras.layers.Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = KL.Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = KL.Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        h = KL.Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

    # output
    h = KL.Dense(dims[0], name='decoder_0')(h)

    return tf.keras.models.Model(inputs=x, outputs=h)


def encoder_constructor(dims, act='relu'):
    n_stacks = len(dims) - 1
    #init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    # input
    x = tf.keras.layers.Input(shape=(dims[0],), name='input_encoder')
    h = x
    # internal layers in encoder
    for i in range(n_stacks-1):
        h = tf.keras.layers.Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
    # hidden layer
    h = tf.keras.layers.Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here
    encoder = tf.keras.models.Model(inputs=x, outputs=h, name='encoder')
    return encoder

def decoder_constructor(dims, act='relu'):
    n_stacks = len(dims) - 1
    #init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    # input
    z = tf.keras.layers.Input(shape=(dims[-1],), name='input_decoder')
    y = z
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = tf.keras.layers.Dense(dims[i], activation=act, name='decoder_%d' % i)(y)
    # output
    y = tf.keras.layers.Dense(dims[0], name='decoder_0')(y)
    decoder = tf.keras.models.Model(inputs=z, outputs=y, name='decoder')
    return decoder

def ae_constructor(encoder, decoder, dims):
    x = tf.keras.layers.Input(shape=(dims[0],), name='input_autencoder')
    autoencoder = tf.keras.models.Model(inputs=x, outputs=decoder(encoder(x)), name='autoencoder')
    return autoencoder

