class Config_Finetuning(object):
    """
        Configuration class for fine-tuning machine learning models.

        This class is designed to initialize and store configuration parameters
        needed for the fine-tuning of machine learning models. It includes parameters
        specific to the data, pretrained models, and fine-tuning settings for the PseudoLabel
        algorithm.

        Attributes:
        data_path (str): The file path for the dataset used in fine-tuning.
        MODEL_TYPE (str): The type of model being fine-tuned.
        FILE_NAME (str): Extracted file name from `data_path`, used in naming saved models.
        PRETRAINED_SAVE_DIR (str): The directory path to save or load the pretrained model.

        PSEUDO_N_CLUSTERS (int): The number of clusters to use in the PseudoLabel algorithm.
        PSEUDO_LABEL_RATIO (float): The ratio of pseudo labels to use in the PseudoLabel algorithm.
        ITERATIVE_RATIOS (list of float): The list of ratios to use in the iterative PseudoLabel algorithm.
        SAMPLING_METHOD (str): The sampling method used in the PseudoLabel algorithm.
        DENSITY_FUNCTION (str): The density function used in the PseudoLabel algorithm.
        K_NEAREST_NEIGHBOURS (float): The number of nearest neighbours used in the PseudoLabel algorithm.
        PSEUDO_EPOCHS (int): The number of epochs to use in the PseudoLabel algorithm.
        PSEUDO_BATCH_SIZE (int): The batch size to use in the PseudoLabel algorithm.
        CLASSIFICATION_AUGMENTER (dict): The dictionary of parameters for the classification augmenter.
        PSEUDO_SAVE_DIR (str): The directory path to save or load the fine-tuned model.
    """
    def __init__(self, data_path, model_type):
        """
        The constructor for Config_Finetuning class.

        Initializes the configuration parameters for the fine-tuning process of
        machine learning models. It sets the paths for data, pretrained
        models, and the saved models.
        """

        # Initialize the parent class (object, in this case)
        super(Config_Finetuning, self).__init__()

        # Data and Model Configuration
        self.data_path = data_path
        self.MODEL_TYPE = model_type
        self.FILE_NAME = self.data_path.rpartition('/')[-1][:-4]

        # Load Pretrained Model
        self.PRETRAINED_SAVE_DIR = "/rds/user/mb2315/hpc-work/Data/Saved_Models/" + "Pretrain_NNCLR_" + self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5" 
        
        #("C:/Users/marib/Documents/Github/ML_Spike_Sorting/trained_models/" + "Pretrain_" + self.MODEL_TYPE +
         #                           "_" + self.FILE_NAME + ".h5")

        # PseudoLabels
        self.PSEUDO_N_CLUSTERS = 5
        self.PSEUDO_LABEL_RATIO = None
        self.ITERATIVE_RATIOS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4] 
        self.SAMPLING_METHOD = "weighted"
        self.DENSITY_FUNCTION = "mean"
        self.K_NEAREST_NEIGHBOURS = 0.005
        self.PSEUDO_EPOCHS = 50
        self.PSEUDO_BATCH_SIZE = 128
        self.CLASSIFICATION_AUGMENTER = {"apply_noise": True,
                                         "max_noise_lvl": 0.1,
                                         "scale": (1.0, 1.0),
                                         "name": "classification_augmenter"}

        self.PSEUDO_SAVE_DIR = ("/rds/user/mb2315/hpc-work/Data/Saved_Models/" + "DensityPseudoLabel_NNCLR_" +
                                self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5")
        
        
        #("C:/Users/marib/Documents/Github/ML_Spike_Sorting/trained_models/" + "DensityPseudoLabel_NNCLR_" +
         #                       self.MODEL_TYPE + "_" + self.FILE_NAME + ".h5")
