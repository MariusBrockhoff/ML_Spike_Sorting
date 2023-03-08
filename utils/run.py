# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from filter_signal import *
#from file_opener_raw_recording_data import *
from spike_detection import *
from data_preparation import *
from train_models import *
from model_predict import *
from clustering import *

from config_files.config_file_PerceiverIO import *
from config_files.config_AttnAE_1 import Config_AttnAE_1

from models.PerceiverIO import *
from models.AttnAE_1 import TransformerEncoder_AEDecoder


cwd = os.getcwd()
root_folder = os.sep+"ML_Spike_Sorting"
sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep+"utils"+os.sep)

cwd = os.getcwd()
root_folder = os.sep+"ML_Spike_Sorting"
sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep+"config_files"+os.sep)

cwd = os.getcwd()
root_folder = os.sep+"ML_Spike_Sorting"
sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep+"models"+os.sep)



class Run:
    """Class for running full Spike sorting pipeline"""

    def __init__(self, config):
        self.config = config

    """def extract_spikes(self, self.raw_file_name, self.data_path, self.is_big_file, self.filter_frequencies, self.filtering_method, self.order, self.save_path, self.min_TH, self.dat_points_pre_min, self.dat_points_post_min, self.max_TH, self.chunck_len, self.refrec_period, self.reject_channels, self.file_name, self.save):
        recording_data, electrode_stream, fsample = file_opener_raw_recording_data(self.raw_file_name, self.data_path, self.is_big_file=False)
        filtered_signal = filter_signal(recording_data, self.filter_frequencies, fsample, self.filtering_method="Butter_bandpass", self.order=2)
        spike_file = spike_detection(filtered_signal, electrode_stream, fsample, self.save_path, self.min_TH, self.dat_points_pre_min=20, self.dat_points_post_min=44, self.max_TH=-30, self.chunck_len=300, self.refrec_period=0.002, self.reject_channels=[None], self.file_name=None, self.save=True):
    """

    def prepare_data(self):
        dataset, dataset_test = data_preparation(self.config , self.config.DATA_SAVE_PATH, self.config.DATA_PREP_METHOD,
                                                 self.config.DATA_NORMALIZATION, self.config.TRAIN_TEST_SPLIT,
                                                 self.config.BATCH_SIZE)
        return dataset, dataset_test

    def initialize_model(self):
        print("Initializing model: ", self.config.MODEL_TYPE)
        if self.config.MODEL_TYPE == "AutoPerceiver":
            model = AutoPerceiver(Embedding_dim=self.config.EMBEDDING_DIM,
                      seq_len=self.config.SEQ_LEN,
                      ENC_number_of_layers=self.config.ENC_NUMBER_OF_LAYERS,
                      ENC_state_index=self.config.ENC_STATE_INDEX,
                      ENC_state_channels=self.config.ENC_STATE_CHANNELS,
                      ENC_dff=self.config.ENC_DFF,
                      ENC_x_attn_dim=self.config.ENC_X_ATTN_DIM,
                      ENC_x_attn_heads=self.config.ENC_X_ATTN_HEADS,
                      ENC_depth=self.config.ENC_DEPTH,
                      ENC_attn_dim=self.config.ENC_SELF_ATTN_DIM,
                      ENC_attn_heads=self.config.ENC_NUM_ATTN_HEADS,
                      ENC_dropout_rate=self.config.ENC_DROPOUT_RATE,
                      DEC_number_of_layers=self.config.DEC_NUMBER_OF_LAYERS,
                      DEC_state_index=self.config.DEC_STATE_INDEX,
                      DEC_state_channels=self.config.DEC_STATE_CHANNELS,
                      DEC_dff=self.config.DEC_DFF,
                      DEC_x_attn_dim=self.config.DEC_X_ATTN_DIM,
                      DEC_x_attn_heads=self.config.DEC_X_ATTN_HEADS,
                      DEC_depth=self.config.DEC_DEPTH,
                      DEC_attn_dim=self.config.DEC_SELF_ATTN_DIM,
                      DEC_attn_heads=self.config.DEC_NUM_ATTN_HEADS,
                      DEC_dropout_rate=self.config.DEC_DROPOUT_RATE)

        elif self.config.MODEL_TYPE == "AttnAE_1":
            model_init = TransformerEncoder_AEDecoder(data_prep=self.config.DATA_PREP,
                                                 num_layers=self.config.NUM_LAYERS,
                                                 d_model=self.config.D_MODEL,
                                                 num_heads=self.config.NUM_HEADS,
                                                 dff=self.config.DFF,
                                                 pe_input=self.config.SEQ_LEN,
                                                 dropout=self.config.DROPOUT_RATE,
                                                 dec_dims=self.config.DEC_DIMS)

            return model_init

        
    def train(self, model, dataset, dataset_test):
        loss_lst, test_loss_lst = train_model(model=model, config=self.config, dataset=dataset,
                                              dataset_test=dataset_test, save_weights=self.config.SAVE_WEIGHTS)
        return loss_lst, test_loss_lst
        

    def predict(self, model, dataset, dataset_test):
        encoded_data, encoded_data_test = model_predict_latents(model=model, dataset=dataset, dataset_test=dataset_test)
        return encoded_data, encoded_data_test

    def cluster_data(self, encoded_data, encoded_data_test):
        y_pred, n_clusters = clustering(data=encoded_data, method=self.config.CLUSTERING_METHOD, n_clusters=self.config.N_CLUSTERS,
                                        eps=self.config.EPS, min_cluster_size=self.config.MIN_CLUSTER_SIZE)
        y_pred_test, n_clusters_test = clustering(data=encoded_data_test, method=self.config.CLUSTERING_METHOD,
                                                  n_clusters=self.config.N_CLUSTERS, eps=self.config.EPS,
                                                  min_cluster_size=self.config.MIN_CLUSTER_SIZE)
        return y_pred, n_clusters, y_pred_test, n_clusters_test


    #def evaluate_spike_sorting(self, ):
        #accuracy and other metrices


###DATA

DATA_SAVE_PATH = '/Users/jakobtraeuble/PycharmProjects/ML_Spike_Sorting/spikes_test/Small_SpikesFile_1.pkl'
DATA_PREP_METHOD = "gradient"
DATA_NORMALIZATION = "Standard"
TRAIN_TEST_SPLIT = 0.1

###GENERAL

DROPOUT_RATE = 0.1
NUM_EPOCHS = 100
PLOT = False
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WITH_WARMUP = True
LR_WARMUP = 0.0001
WITH_WD = True
WEIGHT_DECAY = 0.0001
SAVE_WEIGHTS = False


### DATA PREPROCESSING

DATA_PREP = 'embedding'
D_MODEL = 128
DEC_DIMS = [128, 32, 8, 3]
NUM_LAYERS = 8
DFF = 512
NUM_HEADS = 8


Config_AttnAE_1 = Config_AttnAE_1(data_save_path=DATA_SAVE_PATH,
                                  data_prep_method=DATA_PREP_METHOD,
                                  data_normalization=DATA_NORMALIZATION,
                                  train_test_split=TRAIN_TEST_SPLIT,
                                  data_prep=DATA_PREP,
                                  num_layers=NUM_LAYERS,
                                  d_model=D_MODEL,
                                  dff=DFF,
                                  num_heads=NUM_HEADS,
                                  dropout_rate=DROPOUT_RATE,
                                  dec_dims=DEC_DIMS,
                                  num_epochs=NUM_EPOCHS,
                                  plot=PLOT,
                                  batch_size=BATCH_SIZE,
                                  learning_rate=LEARNING_RATE,
                                  with_warmup=WITH_WARMUP,
                                  lr_warmup=LR_WARMUP,
                                  with_wd=WITH_WD,
                                  weight_decay=WEIGHT_DECAY,
                                  save_weights=SAVE_WEIGHTS)


run = Run(Config_AttnAE_1)
dataset, dataset_test = run.prepare_data()
model = run.initialize_model()
loss_lst, test_loss_lst = run.train(model=model, dataset=dataset, dataset_test=dataset_test)
encoded_data, encoded_data_test = run.predict()
y_pred, n_clusters, y_pred_test, n_clusters_test = run.cluster_data()


