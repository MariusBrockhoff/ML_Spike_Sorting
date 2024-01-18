from models.DenseAutoencoder import *


def model_initializer(config):
    """
    Initialize and return the specified machine learning model based on the provided configuration.

    This function supports the initialization of different types of models. Currently, it is set up
    to initialize a 'DenseAutoencoder' model. The function can be extended to support more models.

    Arguments:
        config: A configuration object containing model specifications. It should have attributes like
                MODEL_TYPE, DIMS, ACT, etc., which are used to specify the model and its parameters.

    Returns:
        model: An instance of the specified model. Currently, it returns an instance of DenseAutoencoder.

    Raises:
        ValueError: If an invalid model type is specified in the configuration.
    """
    print('---' * 30)
    print('INITIALIZING MODEL...')

    if config.MODEL_TYPE == "DenseAutoencoder":
        model = DenseAutoencoder(dims=config.DIMS,
                                 act=config.ACT)
    else:
        raise ValueError("please choose a valid Model Type. See Documentation!")

    return model