# -*- coding: utf-8 -*-
class Config_DenseAutoencoder(object):

    def __init__(self):
        super(Config_DenseAutoencoder, self).__init__()

        # Architecture

        self.MODEL_TYPE = "DenseAutoencoder"

        self.SEQ_LEN = 63

        self.LATENT_LEN = 10

        self.DIMS = [self.SEQ_LEN, 500, 500, 2000, self.LATENT_LEN]

        self.ACT = "relu"


