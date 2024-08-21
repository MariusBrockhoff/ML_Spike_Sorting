class Config_Pretraining(object):
    """
     Configuration class for the pretraining phase of machine learning models.

     This class initializes and stores configuration parameters necessary for
     the pretraining of machine learning models. It includes settings for data
     preparation, model training, and clustering, tailored to the specific requirements
     of the model and dataset.

     Attributes:
         data_path (str): The file path for the dataset used in pretraining.
         MODEL_TYPE (str): The type of model being pretrained.
         FILE_NAME (str): Extracted file name from `data_path`, used for saving the model.

         # Data Preparation Parameters
         DATA_SAVE_PATH (str): The path where processed data will be saved.
         DATA_PREP_METHOD (str): Method used for data preparation, set to "gradient".
         DATA_NORMALIZATION (str): Type of data normalization to apply, set to "MinMax".
         TRAIN_TEST_SPLIT (float): The ratio of train to test data split.
         BENCHMARK_START_IDX (int): The starting index for benchmarking.
         BENCHMARK_END_IDX (int): The ending index for benchmarking, based on the train-test split ratio.

        # NNCLR Parameters
        LEARNING_RATE_NNCLR (float): The initial learning rate for model training.
        WITH_WARMUP_NNCLR (bool): Flag to determine if learning rate warmup should be used.
        LR_WARMUP_NNCLR (int): Number of epochs for learning rate warmup.
        LR_FINAL_NNCLR (float): Final learning rate after warmup.
        NUM_EPOCHS_NNCLR (int): Total number of epochs for training.
        BATCH_SIZE_NNCLR (int): Batch size used for training.
        TEMPERATURE (float): Temperature parameter for NNCLR.
        QUEUE_SIZE (float): Queue size parameter for NNCLR.
        PROJECTION_WIDTH (int): Projection width parameter for NNCLR.
        CONTRASTIVE_AUGMENTER (dict): The dictionary of parameters for the contrastive augmenter.
        CLASSIFICATION_AUGMENTER (dict): The dictionary of parameters for the classification augmenter.

        # Clustering Parameters
        CLUSTERING_METHOD (str): Clustering algorithm to use, set to "Kmeans".
        N_CLUSTERS (int): Number of clusters to form in clustering.
        EPS (float or None): Epsilon parameter for clustering algorithms, if applicable.
        MIN_CLUSTER_SIZE (int): Minimum size for a cluster.
        KNN (int): Number of nearest neighbors to consider in clustering algorithms.

        # Model Saving Parameters
        SAVE_WEIGHTS (bool): Whether to save the model weights after training.
        SAVE_DIR (str): The directory path to save the pretrained model.
    """

    def __init__(self, data_path, model_type):

        # Initialize the parent class (object, in this case)
        super(Config_Pretraining, self).__init__()

        # Data and Model Configuration
        self.data_path = data_path
        self.MODEL_TYPE = model_type
        self.FILE_NAME = self.data_path.rpartition('/')[-1][:-4]

        # Data Preparation Configuration
        self.DATA_SAVE_PATH = self.data_path
        self.DATA_PREP_METHOD = "gradient"
        self.DATA_NORMALIZATION = "MinMax"
        self.TRAIN_TEST_SPLIT = 0.75
        self.BENCHMARK_START_IDX = 0
        self.BENCHMARK_END_IDX = 5  # int(1/self.TRAIN_TEST_SPLIT)


        # NNCLR Configuration
        self.LEARNING_RATE_NNCLR = 1e-3
        self.WITH_WARMUP_NNCLR = False
        self.LR_WARMUP_NNCLR = 10  # 2 #10
        self.LR_FINAL_NNCLR = 1e-4  # 1e-6 1e-8

        self.NUM_EPOCHS_NNCLR = 25
        self.BATCH_SIZE_NNCLR = 256

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

        # Clustering Configuration
        self.CLUSTERING_METHOD = "Kmeans"
        self.N_CLUSTERS = 5
        self.EPS = None
        self.MIN_CLUSTER_SIZE = 1000
        self.KNN = 1000

        # Model Saving Configuration
        self.SAVE_WEIGHTS = True
        self.SAVE_DIR = ("/rds/user/mb2315/hpc-work/Data/Saved_Models/" + "Pretrain_" + self.MODEL_TYPE + "_"
                         + self.FILE_NAME + ".h5") 
        
        #("C:/Users/marib/Documents/Github/ML_Spike_Sorting/trained_models/" + "Pretrain_" + self.MODEL_TYPE + "_"
         #                + self.FILE_NAME + ".h5")
