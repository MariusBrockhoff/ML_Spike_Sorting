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
                      ENC_latent_len=config.ENC_LATENT_LEN,
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
        if self.model == "AutoPerceiver":
            lr = config.LEARNING_RATE
            wd = config.WEIGHT_DECAY
            
            if WITH_WD:
                optimizer = tfa.optimizers.AdamW(weight_decay=wd, learning_rate=lr)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            
            
            #subdir = datetime.datetime.now().strftime('run%Y%m%dT%H%M')
            #chk_dir = config.CHK_DIR + './checkpoints/'+subdir+'/' 
            
            
            for epoch in range(config.NUM_EPOCHS):
                    
                if WITH_WARMUP:
                    lr = config.LEARNING_RATE*min(epoch,config.LR_WARMUP)/config.LR_WARMUP
                    if epoch>config.LR_WARMUP:
                        lr = config.LEARNING_RATE
                        lr = config.LR_FINAL + .5*(lr-config.LR_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
                        wd = wd + .5*(wd-config.WD_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
                else:
                    lr = config.LR_FINAL + .5*(lr-config.LR_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
                    wd = wd + .5*(wd-config.WD_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
            
                
                optimizer.learning_rate = lr
                optimizer.weight_decay = wd
                
                for step,batch in enumerate(dataset):      
                    batch_s = batch
                    with tf.GradientTape() as tape:                   
                      [ENC_state, logits, output] = autoencoder(batch_s)
                      mse = tf.keras.losses.MeanSquaredError()
                      loss = mse(batch_s, output)
                      grads = tape.gradient(loss,autoencoder.trainable_weights)
                      optimizer.apply_gradients(zip(grads,autoencoder.trainable_weights))
        
    #def clustering(self, ) Write full clustering class