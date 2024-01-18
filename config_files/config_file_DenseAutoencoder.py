# -*- coding: utf-8 -*-
# This Python file uses the following encoding: utf-8. This is particularly relevant for Python 2.x to handle Unicode
# characters.
# In Python 3.x, utf-8 is the default encoding.

class Config_DenseAutoencoder(object):
    """
    Configuration class for a Dense Autoencoder.

    This class initializes and stores the configuration parameters
    for a dense (fully connected) autoencoder model. It includes
    parameters for the architecture of the autoencoder such as the
    sequence length, latent dimension length, dimensions of each layer,
    and the activation function used.

    Attributes:
        MODEL_TYPE (str): A string indicating the type of the model, set to 'DenseAutoencoder'.
        SEQ_LEN (int): The length of the input sequences that the autoencoder will process.
        LATENT_LEN (int): The size of the latent space representation in the autoencoder.
        DIMS (list of int): List representing the dimensions of each layer in the autoencoder,
                            including the input layer (SEQ_LEN), intermediate layers (500, 500, 2000),
                            and the latent layer (LATENT_LEN).
        ACT (str): The activation function used in the autoencoder, set to 'relu' (Rectified Linear Unit).
    """

    def __init__(self):
        """
        The constructor for Config_DenseAutoencoder class.

        Initializes the configuration parameters for the autoencoder model.
        The super() method is called to initialize the parent class,
        which is object in this case, as Config_DenseAutoencoder doesn't explicitly inherit from another class.
        """

        # Initialize the parent class (object, in this case)
        super(Config_DenseAutoencoder, self).__init__()

        # Model Type
        self.MODEL_TYPE = "DenseAutoencoder"

        # Input Sequence Length
        self.SEQ_LEN = 63

        # Latent Space Length
        self.LATENT_LEN = 10

        # Layer Dimensions
        self.DIMS = [self.SEQ_LEN, 500, 500, 2000, self.LATENT_LEN]

        # Activation Function
        self.ACT = "relu"
