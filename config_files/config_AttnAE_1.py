# -*- coding: utf-8 -*-
class Config_AttnAE_1(object):

    def __init__(self):
        super(Config_AttnAE_1, self).__init__()

        self.MODEL_TYPE = "AttnAE_1"

        ###DATA
        self.DATA_SAVE_PATH = '/Users/jakobtraeuble/PycharmProjects/ML_Spike_Sorting/spikes_test/Small_SpikesFile_1.pkl'
        self.DATA_PREP_METHOD = "gradient"
        self.DATA_NORMALIZATION = "Standard"
        self.TRAIN_TEST_SPLIT = 0.1

        ###GENERAL
        self.DROPOUT_RATE = 0.1
        self.NUM_EPOCHS = 100
        self.PLOT = False
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.001
        self.WITH_WARMUP = True
        self.LR_WARMUP = 0.0001
        self.WITH_WD = True
        self.WEIGHT_DECAY = 0.0001
        self.SAVE_WEIGHTS = False

        ### MODEL ARCHITECTURE
        self.DATA_PREP = 'embedding'

        # num_layers = number of attention modules
        self.NUM_LAYERS = 8

        # dff = shape of dense layer in attention module
        self.DFF = 512

        # num_heads = (assert d_model//num_heads)
        self.NUM_HEADS = 8

        # dec_dims: list of dimensions of dense decoder
        self.DEC_DIMS = [128, 32, 8, 3]

        # d_model: embedding
        self.D_MODEL = 32


