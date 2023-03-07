# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

cwd = os.getcwd()

root_folder = os.sep+"ML_Spike_Sorting"
sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep+"utils"+os.sep)

from filter_signal import *
from file_opener_raw_recording_data import *
from spike_detection import *
from data_preparation import *
from clustering import *

#Do imports of models, config files etc



class Run:
	"""Class for running full Spike sorting pipeline"""
	
	def __init__(self, raw_file_name, data_path, is_big_file, filter_frequencies, filtering_method, order, save_path, min_TH, dat_points_pre_min, dat_points_post_min, max_TH, chunck_len, refrec_period, reject_channels, file_name, save):
		"""
			Args:
				hfpn: instance of HFPN after all runs have been completed
				save_all_data (bool): whether to save every data point of the time-series data
		"""

		self.raw_file_name = raw_file_name
		self.data_path = data_path
		self.is_big_file = is_big_file
		self.filter_frequencies = filter_frequencies
		self.filtering_method = filtering_method
		self.order = order
		self.save_path = save_path
        self.min_TH = min_TH
		self.dat_points_pre_min = dat_points_pre_min
		self.dat_points_post_min = dat_points_post_min
		self.max_TH = max_TH
		self.chunck_len = chunck_len
		self.refrec_period = refrec_period
		self.reject_channels = reject_channels
		self.file_name = file_name
		self.save = save
        
        self.data_prep_method = data_prep_method
        self.normalization = normalization
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        
        self.model = model

	def extract_spikes(self, self.raw_file_name, self.data_path, self.is_big_file, self.filter_frequencies, self.filtering_method, self.order, self.save_path, self.min_TH, self.dat_points_pre_min, self.dat_points_post_min, self.max_TH, self.chunck_len, self.refrec_period, self.reject_channels, self.file_name, self.save):
		"""Takes raw recording and produces spike file"""
        recording_data, electrode_stream, fsample = file_opener_raw_recording_data(self.raw_file_name, self.data_path, self.is_big_file=False)
        filtered_signal = filter_signal(recording_data, self.filter_frequencies, fsample, self.filtering_method="Butter_bandpass", self.order=2)
        spike_file = spike_detection(filtered_signal, electrode_stream, fsample, self.save_path, self.min_TH, self.dat_points_pre_min=20, self.dat_points_post_min=44, self.max_TH=-30, self.chunck_len=300, self.refrec_period=0.002, self.reject_channels=[None], self.file_name=None, self.save=True):
            
    
    #def load_config_file(self, ):
    
    def prepare_data(self, self.save_path, self.data_prep_method, self.normalization, self.train_test_split, self.batch_size):
        dataset, dataset_test = data_praparation(self.save_path, self.data_prep_method, self.normalization, self.train_test_split, self.batch_size)
        return dataset, dataset_test
    
    
    def initialize_model(self, self.model):
        if self.model == "AutoPerceiver":
            config = Config_AutoPerceiver
            autoencoder = AutoPerceiver(Embedding_dim=config.EMBEDDING_DIM,
                      seq_len=config.SEQ_LEN,
                      ENC_number_of_layers=config.ENC_NUMBER_OF_LAYERS,
                      ENC_state_index=config.ENC_STATE_INDEX,
                      ENC_state_channels=config.ENC_STATE_CHANNELS,                    
                      ENC_dff=config.ENC_DFF,
                      ENC_x_attn_dim=config.ENC_X_ATTN_DIM,
                      ENC_x_attn_heads=config.ENC_X_ATTN_HEADS,
                      ENC_depth=config.ENC_DEPTH,     
                      ENC_attn_dim=config.ENC_SELF_ATTN_DIM,
                      ENC_attn_heads=config.ENC_NUM_ATTN_HEADS,
                      ENC_dropout_rate=config.ENC_DROPOUT_RATE,
                      DEC_number_of_layers=config.DEC_NUMBER_OF_LAYERS,
                      DEC_state_index=config.DEC_STATE_INDEX,
                      DEC_state_channels=config.DEC_STATE_CHANNELS,                     
                      DEC_dff=config.DEC_DFF,
                      DEC_x_attn_dim=config.DEC_X_ATTN_DIM,
                      DEC_x_attn_heads=config.DEC_X_ATTN_HEADS,
                      DEC_depth=config.DEC_DEPTH,     
                      DEC_attn_dim=config.DEC_SELF_ATTN_DIM,
                      DEC_attn_heads=config.DEC_NUM_ATTN_HEADS,
                      DEC_dropout_rate=config.DEC_DROPOUT_RATE)
        
    def train_model(self, self.model):
        train_model(self.model, save_weights)
        
        
    
    def cluster_data(self, ):
        y_pred, n_clusters = clustering(data=, method=config.CLUSTERING_METHOD, n_clusters=config.N_CLUSTERS, eps=config.EPS, min_cluster_size=config.MIN_CLUSTER_SIZE)