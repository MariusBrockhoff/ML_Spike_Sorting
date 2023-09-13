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

        self.BENCHMARK_END_IDX = 5 #int(1/self.TRAIN_TEST_SPLIT)


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

        self.PATIENCE = 10

        self.MIN_DELTA = 0

        self.BASELINE = 0.0001


        #Architecture

        self.MODEL_TYPE = "PerceiverIO"

        self.EMBEDDING_DIM = 256

        self.SEQ_LEN = 63

        self.LATENT_LEN = 10


        # Encoder

        self.ENC_NUMBER_OF_LAYERS = 10

        self.ENC_STATE_INDEX = 32

        self.ENC_STATE_CHANNELS = 256

        self.ENC_DFF = self.ENC_STATE_CHANNELS*2

        self.ENC_X_ATTN_HEADS = 1

        self.ENC_X_ATTN_DIM = 256 #int(self.ENC_STATE_CHANNELS / self.ENC_X_ATTN_HEADS)

        self.ENC_DEPTH = 4

        self.ENC_NUM_ATTN_HEADS = 8

        self.ENC_SELF_ATTN_DIM = 32 #int(self.ENC_STATE_CHANNELS / self.ENC_NUM_ATTN_HEADS)

        self.ENC_DROPOUT_RATE = 0.1


        #Decoder

        self.DEC_NUMBER_OF_LAYERS = 10

        self.DEC_STATE_INDEX = 32

        self.DEC_STATE_CHANNELS = 256

        self.DEC_DFF = self.DEC_STATE_CHANNELS*2

        self.DEC_X_ATTN_HEADS = 1

        self.DEC_X_ATTN_DIM = 256 #int(self.DEC_STATE_CHANNELS / self.DEC_X_ATTN_HEADS)

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

        self.SAVE_DIR = self.SAVE_DIR = "/rds/user/mb2315/hpc-work/Data/Saved_Models/" + "Pretrain_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5"

        #self.SAVE_WEIGHTS = False

        #self.SAVE_DIR = '/home/mb2315/ML_Spike_Sorting/trained_models/{MODEL_TYPE}_{DATA_PREP_METHOD}_{DATA_NORMALIZATION}' \
         #               '_{DATA_AUG}_{LEARNING_RATE}_{LR_FINAL}_{BATCH_SIZE}_{EMBEDDING_DIM}_{LATENT_LEN}_{ENC_NUMBER_OF_LAYERS}' \
          #              '_{ENC_STATE_INDEX}_{ENC_STATE_CHANNELS}_{ENC_DEPTH}_{ENC_DROPOUT_RATE}_{DEC_NUMBER_OF_LAYERS}' \
           #             '_{DEC_STATE_INDEX}_{DEC_STATE_CHANNELS}_{DEC_DEPTH}_{DEC_DROPOUT_RATE}.pth'.format(
            #MODEL_TYPE=self.MODEL_TYPE,
            #DATA_PREP_METHOD=self.DATA_PREP_METHOD,
            #DATA_NORMALIZATION=self.DATA_NORMALIZATION,
            #DATA_AUG=self.DATA_AUG,
            #LEARNING_RATE=self.LEARNING_RATE,
            #LR_FINAL=self.LR_FINAL,
            #BATCH_SIZE=self.BATCH_SIZE,
            #EMBEDDING_DIM=self.EMBEDDING_DIM,
            #LATENT_LEN=self.LATENT_LEN,
            #ENC_NUMBER_OF_LAYERS=self.ENC_NUMBER_OF_LAYERS,
            #ENC_STATE_INDEX=self.ENC_STATE_INDEX,
            #ENC_STATE_CHANNELS=self.ENC_STATE_CHANNELS,
            #ENC_DEPTH=self.ENC_DEPTH,
            #ENC_DROPOUT_RATE=self.ENC_DROPOUT_RATE,
            #DEC_NUMBER_OF_LAYERS=self.DEC_NUMBER_OF_LAYERS,
            #DEC_STATE_INDEX=self.DEC_STATE_INDEX,
            #DEC_STATE_CHANNELS=self.DEC_STATE_CHANNELS,
            #DEC_DEPTH=self.DEC_DEPTH,
            #DEC_DROPOUT_RATE=self.DEC_DROPOUT_RATE)


