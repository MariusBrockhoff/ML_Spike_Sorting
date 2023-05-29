# -*- coding: utf-8 -*-
class Config_DINO(object):

    def __init__(self, data_path):
        super(Config_DINO, self).__init__()

        # Data
        self.data_path = data_path

        self.DATA_SAVE_PATH = self.data_path

        self.DATA_PREP_METHOD = "gradient"

        self.DATA_NORMALIZATION = "MinMax"

        self.TRAIN_TEST_SPLIT = 0.2

        self.BENCHMARK_START_IDX = 0

        self.BENCHMARK_END_IDX = 5 #int(1/self.TRAIN_TEST_SPLIT)


        #Data Augmentation

        self.DATA_AUG = True

        self.APPLY_NOISE = True

        self.MAX_NOISE_LVL = 0.1

        self.APPLY_FLIP = True

        self.FLIP_PROBABILITY = 0.5

        self.APPLY_HSHIFT = True

        self.MAX_HSHIFT = 63

        self.CROP_PCTS = [0.9, 0.5]

        self.NUMBER_LOCAL_CROPS = 3


        # TRAINING HYPERPARAMETERS

        self.NUM_EPOCHS = 300

        self.BATCH_SIZE = 128

        self.LEARNING_RATE = 2.5e-4 #1e-4 #1e-6

        self.WITH_WARMUP = True

        self.LR_WARMUP = 10 #2 #10

        self.LR_FINAL = 1e-6  #1e-6 1e-8

        self.WITH_WD = False

        self.WEIGHT_DECAY = 1e-2 #1e-5 1e-7

        self.WD_FINAL = 1e-4    #1e-41e-6

        self.CENTERING_RATE = 0.9  # 0.9

        self.LEARNING_MOMENTUM_RATE = 0.997 # 0.997

        self.STUDENT_TEMPERATURE = 0.1

        self.TEACHER_TEMPERATURE = 0.04  # 0.04

        self.TEACHER_TEMPERATURE_FINAL = 0.07  # 0.07

        self.TEACHER_WARMUP = 30


        self.EARLY_STOPPING = False

        self.PATIENCE = 20

        self.MIN_DELTA = 0

        self.BASELINE = 0.0005


        #Architecture

        self.MODEL_TYPE = "DINO"

        self.EMBEDDING_DIM = 256

        self.SEQ_LEN = 63

        self.LATENT_LEN = 100

        self.NUMBER_OF_LAYERS = 3

        self.STATE_INDEX = 32

        self.STATE_CHANNELS = 256

        self.DFF = self.STATE_CHANNELS * 2

        self.X_ATTN_HEADS = 1

        self.X_ATTN_DIM = 64

        self.DEPTH = 4

        self.NUM_ATTN_HEADS = 8

        self.SELF_ATTN_DIM = 64

        self.DROPOUT_RATE = 0


        #Clustering

        self.CLUSTERING_METHOD = "Kmeans"

        self.N_CLUSTERS = 5

        self.EPS = None

        self.MIN_CLUSTER_SIZE = 1000

        self.KNN = 1000

        # SAVE MODEL

        self.LOAD = False

        self.LOAD_DIR = "C:\\Users\\marib\\Documents\\Github\\ML_Spike_Sorting\\Data\\Model_test.tf"

        self.SAVE_WEIGHTS = False

        self.SAVE_DIR = False


