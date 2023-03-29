# -*- coding: utf-8 -*-
class Config_AttnAE_2(object):

    def __init__(self, data_path):
        super(Config_AttnAE_2, self).__init__()
        self.data_path = data_path

        # Data

        self.DATA_SAVE_PATH = self.data_path

        self.DATA_PREP_METHOD = "gradient"

        self.DATA_NORMALIZATION = "Standard"

        self.TRAIN_TEST_SPLIT = 0.1



        # TRAINING HYPERPARAMETERS

        self.LEARNING_RATE = 1e-5 #1e-4 #1e-6

        self.WITH_WARMUP = False

        self.LR_WARMUP =  10 #2 #10

        self.LR_FINAL = 1e-7  #1e-6 1e-8

        self.WITH_WD = False

        self.WEIGHT_DECAY = 1e-2 #1e-5 1e-7

        self.WD_FINAL = 1e-4    #1e-41e-6

        self.NUM_EPOCHS = 75

        self.BATCH_SIZE = 128

        # General

        self.SAVE_WEIGHTS = False

        self.SAVE_DIR = None

        self.LOAD = False

        self.LOAD_DIR = None


        #Architecture

        self.MODEL_TYPE = "AttnAE_2"

        #self.EMBEDDING_DIM = 64

        #self.SEQ_LEN = 64


        # Transformer

        self.DFF = 128

        self.D_MODEL = 128

        self.LATENT_LEN = 30

        self.ENC_DEPTH = 12

        self.ENC_NUM_ATTN_HEADS = 8

        self.Q_FOURIER_NUM_BANDS = int((self.D_MODEL - 1) / 2)

        self.ENC_SELF_ATTN_DIM = int(self.D_MODEL / self.ENC_NUM_ATTN_HEADS)

        self.ENC_DROPOUT_RATE = 0.1

        self.DEC_LAYERS = [2000, 500, 500, 64]

        self.REG_VALUE = 10e-3



        #Clustering

        self.CLUSTERING_METHOD = "Kmeans"

        self.N_CLUSTERS = 5

        self.EPS = None

        self.MIN_CLUSTER_SIZE = 1000