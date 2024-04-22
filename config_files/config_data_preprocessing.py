class Config_Preprocessing(object):
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

        # Reconstruction Parameters
        LEARNING_RATE (float): The initial learning rate for model training.
        WITH_WARMUP (bool): Flag to determine if learning rate warmup should be used.
        LR_WARMUP (int): Number of epochs for learning rate warmup.
        LR_FINAL (float): Final learning rate after warmup.
        NUM_EPOCHS (int): Total number of epochs for training.
        BATCH_SIZE (int): Batch size used for training.
        EARLY_STOPPING (bool): Whether to use early stopping.
        PATIENCE (int): Number of epochs to wait for improvement before early stopping.
        MIN_DELTA (float): Minimum change to quantify an improvement.
        BASELINE (int): Baseline value for training metrics.

        # Regularization Parameters
        WITH_WD (bool): Flag to determine if weight decay regularization is used.
        WEIGHT_DECAY (float): Initial value for weight decay.
        WD_FINAL (float): Final value for weight decay after training.

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

    def __init__(self, data_path):
        """
        The constructor for Config_Pretraining class.

        Initializes the configuration parameters for the pretraining process of
        machine learning models. It sets the paths for data, and defines settings
        for model training, clustering, and saving.

        Parameters:
            data_path (str): The file path for the dataset used in pretraining.
        """

        # Initialize the parent class (object, in this case)
        super(Config_Preprocessing, self).__init__()

        # Data and Model Configuration
        self.DATA_PATH = data_path  # Path to the dataset
        self.FILE_NAME = self.DATA_PATH.rpartition('/')[-1][:-3]  # Extracting file name from the data path

        # Opening raw files
        self.BIG_FILE = False # Is file too big to be opened all at once (RAM limitations)

        # Filtering
        self.FILTERING_METHOD = "Butter_bandpass"
        self.FREQUENCIES = [300, 3000]
        self.ORDER = 2

        # Spike Detection
        self.INTERVAL_LENGTH = 300
        self.REJECT_CHANNELS = []
        self.MIN_TH = -5
        self.MAX_TH = -30
        self.DAT_POINTS_PRE_MIN = 20
        self.DAT_POINTS_POST_MIN = 44
        self.REFREC_PERIOD = 0.002

        # Model Saving Configuration
        self.SAVE_WEIGHTS = True  # Flag to save model weights
        self.SAVE_PATH = "/rds/user/mb2315/hpc-work/Data/Saved_Models/Spike_File_" + self.FILE_NAME + ".pkl" # Directory for saving the model

