# -*- coding: utf-8 -*-
class Config_AttnAE(object):

    def __init__(self, data_path):
        super(Config_AttnAE, self).__init__()
        self.data_path = data_path

        self.MODEL_TYPE = "AttnAE"
        self.LOAD = False
        self.LOAD_DIR = None
        self.EARLY_STOPPING = False
        self.PATIENCE = 15
        self.MIN_DELTA = 0
        self.BASELINE = 0

        ###DATA
        self.DATA_SAVE_PATH = self.data_path
        self.DATA_PREP_METHOD = "gradient"
        self.DATA_NORMALIZATION = "Standard"
        self.TRAIN_TEST_SPLIT = 0.1

        ###GENERAL
        self.REG_VALUE = 10e-3
        self.DROPOUT_RATE = 0.1
        self.NUM_EPOCHS = 10 #75
        self.PLOT = False
        self.BATCH_SIZE = 512
        self.LEARNING_RATE = 1e-5 #1e-9 #1e-12 #1e-5
        self.WITH_WARMUP = False #False
        self.LR_WARMUP = 10
        self.LR_FINAL = 1e-12 #1e-12 #1e-15 #1e-9
        self.WITH_WD = False #False
        self.WD_FINAL = 1e-4 #1e-4
        self.WEIGHT_DECAY = 1e-2 #1e-2

        ### MODEL ARCHITECTURE
        self.DATA_PREP = 'embedding'

        #encoder depth = number of attention modules
        self.ENC_DEPTH = 4 #12

        # dff = shape of dense layer in attention module
        self.DFF = 128 #128

        # num_heads = (assert d_model//num_heads)
        self.NUM_ATTN_HEADS = 16

        # DEC_LAYERS: list of dimensions of dense decoder
        self.DEC_LAYERS = [2000, 2000, 500, 500] #[2000, 500, 500, 64] OR [128, 32, 8, 3]

        # d_model: embedding
        self.D_MODEL = 256 #128

        #latent_len: length of latent space
        self.LATENT_LEN = 30

        # Clustering
        self.CLUSTERING_METHOD = "Kmeans"
        self.N_CLUSTERS = 5
        self.EPS = None
        self.MIN_CLUSTER_SIZE = 1000
        self.KNN = 1000

        #AUGMENTATION
        self.DATA_AUG = False
        self.APPLY_NOISE = False
        self.MAX_NOISE_LVL = 0.1
        self.APPLY_FLIP = False
        self.FLIP_PROBABILITY = 0.5
        self.APPLY_HSHIFT = False
        self.MAX_HSHIFT = None

        #SAVE MODEL
        self.SAVE_WEIGHTS = True
        self.SAVE_DIR = '/home/jnt27/ML_Spike_Sorting/trained_models/test.pth'
        '''self.SAVE_DIR = '/home/jnt27/ML_Spike_Sorting/trained_models/{MODEL_TYPE}_{DATA_PREP_METHOD}_{DATA_NORMALIZATION}_{REG_VALUE}_{DROPOUT_RATE}_{DATA_PREP}_{ENC_DEPTH}_{DFF}_{DEC_LAYERS}_{D_MODEL}_{LATENT_LEN}_{DATA_AUG}.pth'.format(
            MODEL_TYPE=self.MODEL_TYPE,
            DATA_PREP_METHOD=self.DATA_PREP_METHOD,
            DATA_NORMALIZATION=self.DATA_NORMALIZATION,
            REG_VALUE=self.REG_VALUE,
            DROPOUT_RATE=self.DROPOUT_RATE,
            DATA_PREP=self.DATA_PREP,
            ENC_DEPTH=self.ENC_DEPTH,
            DFF=self.DFF,
            DEC_LAYERS=self.DEC_LAYERS,
            D_MODEL=self.D_MODEL,
            LATENT_LEN=self.LATENT_LEN,
            DATA_AUG=self.DATA_AUG)'''


