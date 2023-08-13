import tensorflow as tf
from tensorflow.keras import layers as KL

#TODO: make code consistent to other models
class Encoder(tf.keras.Model):

    def __init__(self,

                 dims=[63,500,500,2000,10],

                 act="relu"):

        super(Encoder, self).__init__()

        self.dims = dims

        self.act = act

        self.encoding = [KL.Dense(self.dims[i + 1], activation=self.act, name='encoder_%d' % i) for i in range(len(self.dims) - 2)]

        self.hidden = KL.Dense(self.dims[-1], name='encoder_%d' % (len(self.dims) - 1))


    def call(self, inputs):

        h = inputs #tf.keras.layers.InputLayer(input_shape=(self.dims[0],))(inputs)

        for i in range(len(self.dims) - 2):

          h = self.encoding[i](h)

        h = self.hidden(h)  # hidden layer, features are extracted from here

        return h


class Decoder(tf.keras.Model):
    def __init__(self,

                 dims=[63, 500, 500, 2000, 10],

                 act="relu"):
        super(Decoder, self).__init__()

        self.dims = dims

        self.act = act

        self.decoding = [KL.Dense(self.dims[i], activation=self.act, name='decoder_%d' % i) for i in range(len(self.dims) - 2, 0, -1)]

        self.out = KL.Dense(self.dims[0], name='decoder_0')

    def call(self, inputs):

        h = inputs #tf.keras.layers.InputLayer(input_shape=(self.dims[-1],))(inputs)

        for j in range(len(self.dims) - 2):

          h = self.decoding[j](h)

        h = self.out(h)

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

