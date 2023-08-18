# -*- coding: utf-8 -*-
class Config_DenseAutoencoder(object):

    def __init__(self, data_path):
        super(Config_DenseAutoencoder, self).__init__()

        self.data_path = data_path
        self.FILE_NAME = self.data_path.rpartition('\\')[-1]


        # Data

        self.DATA_SAVE_PATH = self.data_path

        self.DATA_PREP_METHOD = "gradient"

        self.DATA_NORMALIZATION = "MinMax"

        self.TRAIN_TEST_SPLIT = 0.2

        self.BENCHMARK_START_IDX = 0

        self.BENCHMARK_END_IDX = 5 #int(1/self.TRAIN_TEST_SPLIT)


       # TRAINING HYPERPARAMETERS

        self.LEARNING_RATE = 1e-4 #1e-4 #1e-6

        self.WITH_WARMUP = False

        self.LR_WARMUP = 10 #2 #10

        self.LR_FINAL = 1e-4  #1e-6 1e-8

        self.WITH_WD = False

        self.WEIGHT_DECAY = 1e-2 #1e-5 1e-7

        self.WD_FINAL = 1e-4    #1e-41e-6

        self.NUM_EPOCHS = 10

        self.BATCH_SIZE = 4096

        self.EARLY_STOPPING = True

        self.PATIENCE = 10

        self.MIN_DELTA = 0.0001

        self.BASELINE = 0.0007

        self.DATA_AUG = False


        #Architecture

        self.MODEL_TYPE = "DenseAutoencoder"

        self.SEQ_LEN = 63

        self.LATENT_LEN = 10

        self.DIMS = [self.SEQ_LEN, 500, 500, 2000, self.LATENT_LEN]

        self.ACT = "relu"


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

        self.SAVE_DIR = "/rds/user/jnt27/hpc-work/SpikeSorting/trained_models/" + "Pretrained_dense.h5"

        #self.SAVE_DIR = '/home/mb2315/ML_Spike_Sorting/trained_models/{MODEL_TYPE}_{DATA_PREP_METHOD}_{DATA_NORMALIZATION}' \
        #                '_{LEARNING_RATE}_{LR_FINAL}_{BATCH_SIZE}_{LATENT_LEN}_{DIMS}.pth'.format(
        #    MODEL_TYPE=self.MODEL_TYPE,
        #    DATA_PREP_METHOD=self.DATA_PREP_METHOD,
        #    DATA_NORMALIZATION=self.DATA_NORMALIZATION,
        #    LEARNING_RATE=self.LEARNING_RATE,
        #    LR_FINAL=self.LR_FINAL,
        #    BATCH_SIZE=self.BATCH_SIZE,
        #    LATENT_LEN=self.LATENT_LEN,
        #    DIMS=self.DIMS)


