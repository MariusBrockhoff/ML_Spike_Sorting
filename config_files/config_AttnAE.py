# -*- coding: utf-8 -*-
class Config_AttnAE(object):

    def __init__(self, data_path):
        super(Config_AttnAE, self).__init__()
        self.data_path = data_path

        self.MODEL_TYPE = "AttnAE"
        self.SAVE_DIR = None
        self.LOAD = False
        self.LOAD_DIR = None

        ###DATA
        self.DATA_SAVE_PATH = self.data_path
        self.DATA_PREP_METHOD = "gradient"
        self.DATA_NORMALIZATION = "Standard"
        self.TRAIN_TEST_SPLIT = 0.1

        ###GENERAL
        self.REG_VALUE = 10e-3
        self.DROPOUT_RATE = 0.1
        self.NUM_EPOCHS = 100 #75
        self.PLOT = False
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.001 #1e-5
        self.WITH_WARMUP = True #False
        self.LR_WARMUP = 10
        self.LR_FINAL = 1e-7
        self.WITH_WD = True #False
        self.WD_FINAL = 1e-4 #1e-4
        self.WEIGHT_DECAY = 0.0001 #1e-2
        self.SAVE_WEIGHTS = False

        ### MODEL ARCHITECTURE
        self.DATA_PREP = 'embedding'

        #encoder depth = number of attention modules
        self.ENC_DEPTH = 8 #12

        # dff = shape of dense layer in attention module
        self.DFF = 512 #128

        # num_heads = (assert d_model//num_heads)
        self.NUM_ATTN_HEADS = 8

        # DEC_LAYERS: list of dimensions of dense decoder
        self.DEC_LAYERS = [128, 32, 8, 3] #[2000, 500, 500, 64]

        # d_model: embedding
        self.D_MODEL = 32 #128

        #latent_len: length of latent space
        self.LATENT_LEN = 30

        # Clustering

        self.CLUSTERING_METHOD = "Kmeans"
        self.N_CLUSTERS = 5
        self.EPS = None
        self.MIN_CLUSTER_SIZE = 1000


