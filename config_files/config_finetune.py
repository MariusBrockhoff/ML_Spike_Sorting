#TODO: split original DEC config file (and IDEC, etc accordingly) into pretrain/model config files and finetune config files

class Config_Finetuning(object):

    def __init__(self, data_path):
        super(Config_Finetuning, self).__init__()

        self.data_path = data_path
        self.FILE_NAME = self.data_path.rpartition('\\')[-1]


        # Load Pretrained Model

        self.PRETRAINED_SAVE_DIR = "C:/Users/marib/Documents/Github/ML_Spike_Sorting/trained_models/" + "Pretrained_Perceiver_"+ self.FILE_NAME + ".h5"

        # DEC

        self.DEC_N_CLUSTERS= 5

        self.DEC_BATCH_SIZE = 256

        self.DEC_LEARNING_RATE = 0.01

        self.DEC_MOMENTUM = 0.9

        self.DEC_TOL = 0.001

        self.DEC_MAXITER = 1000

        self.DEC_UPDATE_INTERVAL = 140

        self.DEC_SAVE_DIR = "C:/Users/marib/Documents/Github/ML_Spike_Sorting/trained_models/" + "DEC_"+ self.FILE_NAME + ".h5"

        # IDEC

        self.IDEC_N_CLUSTERS = 5

        self.IDEC_BATCH_SIZE = 256

        self.IDEC_LEARNING_RATE = 0.1

        self.IDEC_MOMENTUM = 0.99

        self.IDEC_GAMMA = 0.1

        self.IDEC_TOL = 0.001

        self.IDEC_MAXITER = 1000

        self.IDEC_UPDATE_INTERVAL = 140

        self.IDEC_SAVE_DIR = "C:/Users/marib/Documents/Github/ML_Spike_Sorting/trained_models/" + "IDEC_"+ self.FILE_NAME + ".h5"


