# -*- coding: utf-8 -*-
class Config_PerceiverIO(object):

    def __init__(self, data_path):
        super(Config_PerceiverIO, self).__init__()
        
        self.data_path = data_path
        self.FILE_NAME = self.data_path.rpartition('/')[-1][:-4]



        # Data

        self.DATA_SAVE_PATH = self.data_path

        self.DATA_PREP_METHOD = "gradient"

        self.DATA_NORMALIZATION = "MinMax"

        self.TRAIN_TEST_SPLIT = 0.2

        self.BENCHMARK_START_IDX = 0

        self.BENCHMARK_END_IDX = 5


       # TRAINING HYPERPARAMETERS

        self.LEARNING_RATE = 2e-5 #1e-4 #1e-6

        self.WITH_WARMUP = True

        self.LR_WARMUP = 10 #2 #10

        self.LR_FINAL = 1e-9  #1e-6 1e-8

        self.WITH_WD = False

        self.WEIGHT_DECAY = 1e-2 #1e-5 1e-7

        self.WD_FINAL = 1e-4    #1e-41e-6

        self.NUM_EPOCHS = 100

        self.BATCH_SIZE = 512

        self.EARLY_STOPPING = True

        self.PATIENCE = 100

        self.MIN_DELTA = 0

        self.BASELINE = 0.001

        #NNCLR




        #Architecture

        self.MODEL_TYPE = "PerceiverIO"

        self.EMBEDDING_DIM = 256

        self.SEQ_LEN = 63

        self.LATENT_LEN = 10


        # Encoder

        self.ENC_NUMBER_OF_LAYERS = 5

        self.ENC_STATE_INDEX = 32

        self.ENC_STATE_CHANNELS = 256

        self.ENC_DFF = self.ENC_STATE_CHANNELS*4

        self.ENC_X_ATTN_HEADS = 8

        self.ENC_X_ATTN_DIM = 32 #int(self.ENC_STATE_CHANNELS / self.ENC_X_ATTN_HEADS)

        self.ENC_DEPTH = 4

        self.ENC_NUM_ATTN_HEADS = 8

        self.ENC_SELF_ATTN_DIM = 32 #int(self.ENC_STATE_CHANNELS / self.ENC_NUM_ATTN_HEADS)

        self.ENC_DROPOUT_RATE = 0.1


        #Decoder

        self.DEC_NUMBER_OF_LAYERS = 5

        self.DEC_STATE_INDEX = 32

        self.DEC_STATE_CHANNELS = 256

        self.DEC_DFF = self.DEC_STATE_CHANNELS*4

        self.DEC_X_ATTN_HEADS = 8

        self.DEC_X_ATTN_DIM = 32 #int(self.DEC_STATE_CHANNELS / self.DEC_X_ATTN_HEADS)

        self.DEC_DEPTH = 4

        self.DEC_NUM_ATTN_HEADS = 8

        self.DEC_SELF_ATTN_DIM = 32 #int(self.DEC_STATE_CHANNELS / self.DEC_NUM_ATTN_HEADS)

        self.DEC_DROPOUT_RATE = 0

        # Data Augmentation

        self.DATA_AUG = False

        self.APPLY_NOISE = False

        self.MAX_NOISE_LVL = 0.1

        self.APPLY_FLIP = False

        self.FLIP_PROBABILITY = 0.5

        self.APPLY_HSHIFT = False

        self.MAX_HSHIFT = None


        #Clustering

        self.CLUSTERING_METHOD = "Kmeans"

        self.N_CLUSTERS = 5

        self.EPS = None

        self.MIN_CLUSTER_SIZE = 1000

        self.KNN = 1000

        # SAVE MODEL

        self.LOAD = False

        self.LOAD_DIR = "C:\\Users\\marib\\Documents\\Github\\ML_Spike_Sorting\\Data\\Model_test.tf"

        self.SAVE_WEIGHTS = True

        self.SAVE_DIR = "/rds/user/mb2315/hpc-work/Data/Saved_Models/" + "Pretrain_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5"
