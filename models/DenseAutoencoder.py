import tensorflow as tf
from tensorflow.keras import layers as KL


class Encoder(tf.keras.Model):
    """
    Encoder part of an Autoencoder.

    This class defines an encoder model in a neural network, which is typically used
    in an autoencoder architecture for dimensionality reduction or feature extraction.

    Attributes:
        dims (list of int): List of integers where each integer represents the number of neurons
                            in each layer of the encoder.
        act (str): The activation function to use in each layer of the encoder.
        encoding (list of keras.layers.Dense): List of Dense layers that make up the encoder.

    Methods:
        call(inputs): Function to compute the output of the encoder given an input.
    """

    def __init__(self, dims=[63, 500, 500, 2000, 10], act="relu"):
        """
        Initializes the Encoder model with the specified layer dimensions and activation function.
        """
        super(Encoder, self).__init__()
        self.dims = dims
        self.act = act
        # Creates a list of Dense layers for the encoder part, excluding the last dimension.
        self.encoding = [KL.Dense(self.dims[i + 1], activation=self.act, name='encoder_%d' % i) for i in
                         range(len(self.dims) - 2)]
        # The hidden layer representation
        self.hidden = KL.Dense(self.dims[-1], name='encoder_%d' % (len(self.dims) - 1))

    def call(self, inputs):
        """
        Function to compute the output of the encoder given an input.
        Applies each layer in the encoder to the input in sequence.
        """
        h = inputs
        for i in range(len(self.dims) - 2):
            h = self.encoding[i](h)
        h = self.hidden(h)
        return h


class Decoder(tf.keras.Model):
    """
    Decoder part of an Autoencoder.

    This class defines a decoder model in a neural network, typically used
    in conjunction with an encoder in an autoencoder architecture for data
    reconstruction or generative tasks.

    Attributes:
        dims (list of int): List of integers representing the number of neurons
                            in each layer of the decoder.
        act (str): Activation function used in each layer of the decoder.
        decoding (list of keras.layers.Dense): List of Dense layers making up the decoder.

    Methods:
        call(inputs): Function to compute the output of the decoder given an input.
    """
    def __init__(self, dims=[63, 500, 500, 2000, 10], act="relu"):
        """
        Initializes the Decoder model with the specified layer dimensions and activation function.
        """
        super(Decoder, self).__init__()
        self.dims = dims
        self.act = act
        # Creates a list of Dense layers for the decoder part, in reverse order excluding the first dimension.
        self.decoding = [KL.Dense(self.dims[i], activation=self.act,
                                  name='decoder_%d' % i) for i in range(len(self.dims) - 2, 0, -1)]
        # The output layer
        self.out = KL.Dense(self.dims[0], name='decoder_0')

    def call(self, inputs):
        """
        Function to compute the output of the decoder given an input.
        Applies each layer in the decoder to the input in sequence.
        """
        h = inputs
        for j in range(len(self.dims) - 2):
            h = self.decoding[j](h)
        h = self.out(h)
        return h


class DenseAutoencoder(tf.keras.Model):
    """
    Dense Autoencoder model combining both Encoder and Decoder.
    This class defines a complete autoencoder model by integrating the Encoder
    and Decoder. It is used for tasks like dimensionality reduction, feature extraction,
    and generative modeling.

    Attributes:
        dims (list of int): List of integers representing the number of neurons in each layer of the autoencoder.
        act (str): Activation function used in each layer of the autoencoder.
        Encoder (Encoder): The encoder part of the autoencoder.
        Decoder (Decoder): The decoder part of the autoencoder.

    Methods:
        call(inputs): Function to compute the output of the autoencoder given an input.
    """

    def __init__(self, dims=[63, 500, 500, 2000, 10], act="relu"):
        """
        Initializes the Dense Autoencoder model with the specified layer dimensions and activation function.
        """
        super(DenseAutoencoder, self).__init__()
        self.dims = dims
        self.act = act
        self.Encoder = Encoder(dims=self.dims, act=self.act)
        self.Decoder = Decoder(dims=self.dims, act=self.act)

    def call(self, inputs):
        """
        Function to compute the output of the autoencoder given an input.
        Passes the input through the Encoder and then through the Decoder.
        """
        logits = self.Encoder(inputs)
        output = self.Decoder(logits)
        return logits, output


def encoder_constructor(dims, act='relu'):
    """
    Function to construct an encoder model.
    Args:
        dims (list of int): List of dimensions for each layer in the encoder.
        act (str): Activation function for the layers.

    Returns:
        A tf.keras.models.Model representing the encoder.
    """
    n_stacks = len(dims) - 1
    x = tf.keras.layers.Input(shape=(dims[0],), name='input_encoder')
    h = x
    for i in range(n_stacks - 1):
        h = tf.keras.layers.Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
    h = tf.keras.layers.Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
    encoder = tf.keras.models.Model(inputs=x, outputs=h, name='encoder')
    return encoder


def decoder_constructor(dims, act='relu'):
    """
    Function to construct a decoder model.
    Args:
        dims (list of int): List of dimensions for each layer in the decoder.
        act (str): Activation function for the layers.

    Returns:
        A tf.keras.models.Model representing the decoder.
    """
    n_stacks = len(dims) - 1
    z = tf.keras.layers.Input(shape=(dims[-1],), name='input_decoder')
    y = z
    for i in range(n_stacks - 1, 0, -1):
        y = tf.keras.layers.Dense(dims[i], activation=act, name='decoder_%d' % i)(y)
    y = tf.keras.layers.Dense(dims[0], name='decoder_0')(y)
    decoder = tf.keras.models.Model(inputs=z, outputs=y, name='decoder')
    return decoder


def ae_constructor(encoder, decoder, dims):
    """
    Function to construct an autoencoder model.
    Combines a given encoder and decoder into a single autoencoder model. This function
    is useful for creating autoencoders where the encoder and decoder are defined
    separately and then integrated.

    Args:
        encoder (tf.keras.models.Model): The encoder model.
        decoder (tf.keras.models.Model): The decoder model.
        dims (list of int): List of dimensions for each layer in the autoencoder, used for defining the input layer.

    Returns:
        A tf.keras.models.Model representing the complete autoencoder.
    """
    x = tf.keras.layers.Input(shape=(dims[0],), name='input_autencoder')
    autoencoder = tf.keras.models.Model(inputs=x, outputs=decoder(encoder(x)), name='autoencoder')
    return autoencoder
