#TODO: split original DEC config file (and IDEC, etc accordingly) into pretrain/model config files and finetune config files

class Config_Finetuning(object):

    def __init__(self, data_path):
        super(Config_Finetuning, self).__init__()


        # Pretrained Model

        self.PRETRAINED_SAVE_DIR = "/rds/user/mb2315/hpc-work/Data/Saved_Models/PRETRAINED_AE_" + FILE_NAME + ".h5"

        # DEC

        self.DEC_N_CLUSTERS= 5

        self.DEC_BATCH_SIZE = 256

        self.DEC_LEARNING_RATE = 0.01

        self.DEC_MOMENTUM = 0.9

        self.DEC_TOL = 0.001

        self.DEC_MAXITER = 10000

        self.DEC_UPDATE_INTERVAL = 140

        self.DEC_SAVE_DIR = '/rds/user/mb2315/hpc-work/Data/Saved_Models/DEC_' + FILE_NAME

        # IDEC

        self.IDEC_BATCH_SIZE = 256

        self.IDEC_LEARNING_RATE = 0.1

        self.IDEC_MOMENTUM = 0.99

        self.IDEC_GAMMA = 0.1

        self.IDEC_TOL = 0.001

        self.IDEC_MAXITER = 20000

        self.IDEC_UPDATE_INTERVAL = 140

        self.IDEC_SAVE_DIR = '/rds/user/mb2315/hpc-work/Data/Saved_Models/IDEC_' + FILE_NAME



