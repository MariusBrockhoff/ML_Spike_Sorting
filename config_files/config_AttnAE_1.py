# -*- coding: utf-8 -*-
class Config_AttnAE_1(object):

    def __init__(self, data_save_path, data_prep_method, data_normalization, train_test_split, data_prep, num_layers,
                 d_model, dff, num_heads, dropout_rate, dec_dims, num_epochs, plot, batch_size, learning_rate,
                 with_warmup, lr_warmup, with_wd, weight_decay, save_weights):
        super(Config_AttnAE_1, self).__init__()

        self.MODEL_TYPE = "AttnAE_1"

        ###DATA
        self.DATA_SAVE_PATH = data_save_path
        self.DATA_PREP_METHOD = data_prep_method
        self.DATA_NORMALIZATION = data_normalization
        self.TRAIN_TEST_SPLIT = train_test_split

        ###GENERAL
        self.DROPOUT_RATE = dropout_rate
        self.NUM_EPOCHS = num_epochs
        self.PLOT = plot
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.WITH_WARMUP = with_warmup
        self.LR_WARMUP = lr_warmup
        self.WITH_WD = with_wd
        self.WEIGHT_DECAY = weight_decay
        self.SAVE_WEIGHTS = save_weights

        ### MODEL ARCHITECTURE
        self.DATA_PREP = data_prep

        # num_layers = number of attention modules
        self.NUM_LAYERS = num_layers

        # dff = shape of dense layer in attention module
        self.DFF = dff

        # num_heads = (assert d_model//num_heads)
        self.NUM_HEADS = num_heads

        # dec_dims: list of dimensions of dense decoder
        self.DEC_DIMS = dec_dims

        # d_model: embedding
        self.D_MODEL = d_model


