# -*- coding: utf-8 -*-
class Config_AttnAE_1(object):

    ###GENERAL

    # dropout_rate
    DROPOU_RATE = 0.1

    # epochs: int specify number of epochs to train
    EPOCHS = 100

    # plot: boolean plot reconstruction after every 10 epochs
    PLOT = False


    ### DATA PREPROCESSING

    # data_prep =  'embedding', 'perceiver-style', 'broadcasting', 'spectrogram' (string unqual to former two
    # uses broadcasting as well)
    DATA_PREP = 'embedding'


    ### MODEL ARCHITECTURE


    # d_model: embedding dimension
    D_MODEL = 128

    # dec_dims: list of dimensions of dense decoder
    DEC_DIMS = [128, 32, 8, 3]

    # num_layers = number of attention modules
    NUM_LAYERS = 8

    # dff = shape of dense layer in attention module
    DFF = 512

    # num_heads = (assert d_model//num_heads)
    NUM_HEADS = 8

    def __init__(self, data_prep, num_layers, d_model, dff, num_heads, dropout_rate, dec_dims, epochs, plot):
        super(Config_AttnAE_1, self).__init__()

        self.DATA_PREP = data_prep
        self.NUM_LAYERS = num_layers
        self.DFF = dff
        self.NUM_HEADS = num_heads
        self.DROPOUT_RATE = dropout_rate
        self.EPOCHS = epochs
        self.PLOT = plot
        self.DEC_DIMS = dec_dims
        self.D_MODEL = d_model


