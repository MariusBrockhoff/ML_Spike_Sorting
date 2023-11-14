# TODO: split original DEC config file (and IDEC, etc accordingly) into pretrain/model config files and finetune config files

class Config_Pretraining(object):

    def __init__(self, data_path, model_type):
        super(Config_Pretraining, self).__init__()

        self.data_path = data_path
        self.MODEL_TYPE = model_type
        self.FILE_NAME = self.data_path.rpartition('/')[-1][:-4]

        #Data
        self.DATA_SAVE_PATH = self.data_path
        self.DATA_PREP_METHOD = "gradient"
        self.DATA_NORMALIZATION = "MinMax"
        self.TRAIN_TEST_SPLIT = 0.75
        self.BENCHMARK_START_IDX = 0
        self.BENCHMARK_END_IDX = 5  # int(1/self.TRAIN_TEST_SPLIT)

        # Reconstruction
        self.LEARNING_RATE = 1e-3  # 1e-4 #1e-6
        self.WITH_WARMUP = False
        self.LR_WARMUP = 10  # 2 #10
        self.LR_FINAL = 1e-4  # 1e-6 1e-8

        self.NUM_EPOCHS = 100
        self.BATCH_SIZE = 4096
        self.EARLY_STOPPING = True
        self.PATIENCE = 10
        self.MIN_DELTA = 0.0001
        self.BASELINE = 0

        self.WITH_WD = False
        self.WEIGHT_DECAY = 1e-2  # 1e-5 1e-7
        self.WD_FINAL = 1e-4  # 1e-41e-6



        # NNCLR
        self.LEARNING_RATE_NNCLR = 1e-3
        self.WITH_WARMUP_NNCLR = False
        self.LR_WARMUP_NNCLR = 10  # 2 #10
        self.LR_FINAL_NNCLR = 1e-4  # 1e-6 1e-8

        self.NUM_EPOCHS_NNCLR = 25
        self.BATCH_SIZE_NNCLR = 2048

        self.TEMPERATURE = 0.1
        self.QUEUE_SIZE = 0.1
        self.PROJECTION_WIDTH = 10
        self.CONTRASTIVE_AUGMENTER = {"apply_noise": True,
                                 "max_noise_lvl": 0.075,
                                 "scale": (1.0, 1.0),
                                 "name": "contrastive_augmenter"}

        self.CLASSIFICATION_AUGMENTER = {"apply_noise": True,
                                    "max_noise_lvl": 0.04,
                                    "scale": (1.0, 1.0),
                                    "name": "classification_augmenter"}

        # Clustering
        self.CLUSTERING_METHOD = "Kmeans"
        self.N_CLUSTERS = 15
        self.EPS = None
        self.MIN_CLUSTER_SIZE = 1000
        self.KNN = 1000

        # Saving
        self.SAVE_WEIGHTS = True
        self.SAVE_DIR = "/rds/user/mb2315/hpc-work/Data/Saved_Models/" + "Pretrain_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5"

        # "/rds/user/mb2315/hpc-work/Data/Saved_Models/" + "Pretrain_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5"

        # "C:/Users/marib/Documents/Github/ML_Spike_Sorting/trained_models/" + "Pretrain_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5"