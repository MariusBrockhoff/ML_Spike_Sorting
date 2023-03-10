# -*- coding: utf-8 -*-
class Config_PerceiverIO(object):

    def __init__(self):
        super(Config_PerceiverIO, self).__init__()


        # Data

        self.DATA_SAVE_PATH = "C:\\Users\\marib\\Documents\\Github\\ML_Spike_Sorting\\Data\\Small_SpikesFile_1.pkl"

        self.DATA_PREP_METHOD = "gradient"

        self.DATA_NORMALIZATION = "Standard"

        self.TRAIN_TEST_SPLIT = 0.1


        # General

        self.SAVE_WEIGHTS = False

        self.SAVE_DIR = "C:\\Users\\marib\\Documents\\Github\\ML_Spike_Sorting\\Data\\Model_test.tf"

        self.LOAD = False

        self.LOAD_DIR = "C:\\Users\\marib\\Documents\\Github\\ML_Spike_Sorting\\Data\\Model_test.tf"


        # TRAINING HYPERPARAMETERS

        self.LEARNING_RATE = 1e-5 #1e-4 #1e-6

        self.WITH_WARMUP = False

        self.LR_WARMUP = 10 #2 #10

        self.LR_FINAL = 1e-9  #1e-6 1e-8

        self.WITH_WD = False

        self.WEIGHT_DECAY = 1e-2 #1e-5 1e-7

        self.WD_FINAL = 1e-4    #1e-41e-6

        self.NUM_EPOCHS = 100

        self.BATCH_SIZE = 512


        #Architecture

        self.MODEL_TYPE = "AutoPerceiver"

        self.EMBEDDING_DIM = 64

        self.SEQ_LEN = 64


        # Encoder

        self.ENC_NUMBER_OF_LAYERS = 3

        self.ENC_STATE_INDEX = 32

        self.ENC_STATE_CHANNELS = 512

        self.ENC_DFF = self.ENC_STATE_CHANNELS*2

        self.ENC_X_ATTN_HEADS = 8

        self.ENC_X_ATTN_DIM = int(self.ENC_STATE_CHANNELS / self.ENC_X_ATTN_HEADS)

        self.ENC_DEPTH = 4

        self.ENC_NUM_ATTN_HEADS = 8

        self.ENC_SELF_ATTN_DIM = int(self.ENC_STATE_CHANNELS / self.ENC_NUM_ATTN_HEADS)

        self.ENC_DROPOUT_RATE = 0


        #Decoder

        self.DEC_NUMBER_OF_LAYERS = 1

        self.DEC_STATE_INDEX = 32

        self.DEC_STATE_CHANNELS = 512

        self.DEC_DFF = self.DEC_STATE_CHANNELS*2

        self.DEC_X_ATTN_HEADS = 8

        self.DEC_X_ATTN_DIM = int(self.DEC_STATE_CHANNELS / self.DEC_X_ATTN_HEADS)

        self.DEC_DEPTH = 4

        self.DEC_NUM_ATTN_HEADS = 8

        self.DEC_SELF_ATTN_DIM = int(self.DEC_STATE_CHANNELS / self.DEC_NUM_ATTN_HEADS)

        self.DEC_DROPOUT_RATE = 0


        #Clustering

        self.CLUSTERING_METHOD = "Kmeans"

        self.N_CLUSTERS = 5

        self.EPS = None

        self.MIN_CLUSTER_SIZE = 1000
