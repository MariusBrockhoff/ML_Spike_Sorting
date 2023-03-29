# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import argparse
import statistics
import tensorflow as tf
import time

from utils.filter_signal import *
# from utils.file_opener_raw_recording_data import *
from utils.spike_detection import *
from utils.data_preparation import *
from utils.train_models import *
from utils.model_predict import *
from utils.clustering import *
from utils.evaluation import *

from config_files.config_file_PerceiverIO import *
from config_files.config_AttnAE_1 import *
from config_files.config_AttnAE_2 import *

from models.PerceiverIO import *
from models.AttnAE_1 import *
from models.AttnAE_2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--Model', type=str, required=True)
parser.add_argument('--PathData', type=str, required=True)
parser.add_argument('--Benchmark', action='store_true')
args = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print("devices:", tf.config.list_physical_devices(device_type=None))


class Run:
    """Class for running full Spike sorting pipeline"""

    def __init__(self, config, benchmark):
        self.config = config
        self.benchmark = benchmark

    """def extract_spikes(self, self.raw_file_name, self.data_path, self.is_big_file, self.filter_frequencies, self.filtering_method, self.order, self.save_path, self.min_TH, self.dat_points_pre_min, self.dat_points_post_min, self.max_TH, self.chunck_len, self.refrec_period, self.reject_channels, self.file_name, self.save):
        recording_data, electrode_stream, fsample = file_opener_raw_recording_data(self.raw_file_name, self.data_path, self.is_big_file=False)
        filtered_signal = filter_signal(recording_data, self.filter_frequencies, fsample, self.filtering_method="Butter_bandpass", self.order=2)
        spike_file = spike_detection(filtered_signal, electrode_stream, fsample, self.save_path, self.min_TH, self.dat_points_pre_min=20, self.dat_points_post_min=44, self.max_TH=-30, self.chunck_len=300, self.refrec_period=0.002, self.reject_channels=[None], self.file_name=None, self.save=True):
    """

    def prepare_data(self):
        print('---' * 30)
        print('PREPARING DATA...')
        dataset, dataset_test = data_preparation(self.config, self.config.DATA_SAVE_PATH, self.config.DATA_PREP_METHOD,
                                                 self.config.DATA_NORMALIZATION, self.config.TRAIN_TEST_SPLIT,
                                                 self.config.BATCH_SIZE, self.benchmark)
        return dataset, dataset_test

    def initialize_model(self):
        print('---' * 30)
        print('INITIALIZING MODEL...')
        if self.config.MODEL_TYPE == "PerceiverIO":
            model = AutoPerceiver(embedding_dim=self.config.EMBEDDING_DIM,
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
            model = TransformerEncoder_AEDecoder(data_prep=self.config.DATA_PREP,
                                                 num_layers=self.config.NUM_LAYERS,
                                                 d_model=self.config.D_MODEL,
                                                 num_heads=self.config.NUM_HEADS,
                                                 dff=self.config.DFF,
                                                 pe_input=self.config.SEQ_LEN,
                                                 latent_len=self.config.LATENT_LEN,
                                                 dropout=self.config.DROPOUT_RATE,
                                                 dec_dims=self.config.DEC_DIMS,
                                                 reg_value=self.config.REG_VALUE)


        elif self.config.MODEL_TYPE == "AttnAE_2":
            model = Attention_AE(d_model=self.config.D_MODEL,
                                 dff=self.config.DFF,
                                 seq_len=self.config.SEQ_LEN,
                                 latent_len=self.config.LATENT_LEN,
                                 ENC_depth=self.config.ENC_DEPTH,
                                 ENC_attn_dim=self.config.ENC_SELF_ATTN_DIM,
                                 ENC_attn_heads=self.config.ENC_NUM_ATTN_HEADS,
                                 ENC_dropout_rate=self.config.ENC_DROPOUT_RATE,
                                 DEC_layers=self.config.DEC_LAYERS,
                                 reg_value=self.config.REG_VALUE)

        return model

    def train(self, model, dataset, dataset_test):
        print('---' * 30)
        print('TRAINING MODEL...')
        loss_lst, test_loss_lst = train_model(model=model, config=self.config, dataset=dataset,
                                              dataset_test=dataset_test, save_weights=self.config.SAVE_WEIGHTS,
                                              save_dir=self.config.SAVE_DIR)
        return loss_lst, test_loss_lst

    def predict(self, model, dataset, dataset_test):
        encoded_data, encoded_data_test, y_true, y_true_test = model_predict_latents(model=model, dataset=dataset,
                                                                                     dataset_test=dataset_test)
        return encoded_data, encoded_data_test, y_true, y_true_test

    def cluster_data(self, encoded_data, encoded_data_test):
        print('---' * 30)
        print('CLUSTERING...')
        y_pred, n_clusters = clustering(data=encoded_data, method=self.config.CLUSTERING_METHOD,
                                        n_clusters=self.config.N_CLUSTERS,
                                        eps=self.config.EPS, min_cluster_size=self.config.MIN_CLUSTER_SIZE)
        y_pred_test, n_clusters_test = clustering(data=encoded_data_test, method=self.config.CLUSTERING_METHOD,
                                                  n_clusters=self.config.N_CLUSTERS, eps=self.config.EPS,
                                                  min_cluster_size=self.config.MIN_CLUSTER_SIZE)
        return y_pred, n_clusters, y_pred_test, n_clusters_test

    def evaluate_spike_sorting(self, y_pred, y_pred_test, y_true, y_true_test):
        print('---' * 30)
        print('EVALUATE RESULTS...')
        train_acc, test_acc = evaluate_clustering(y_pred, y_pred_test, y_true, y_true_test)
        return train_acc, test_acc

    def execute_run(self):
        if self.benchmark:
            dataset, dataset_test = run.prepare_data()
            train_acc_lst = []
            test_acc_lst = []
            for i in range(len(dataset)):
                start_time = time.time()
                model = run.initialize_model()
                loss_lst, test_loss_lst = run.train(model=model, dataset=dataset[i], dataset_test=dataset_test[i])
                encoded_data, encoded_data_test, y_true, y_true_test = run.predict(model=model, dataset=dataset[i],
                                                                                   dataset_test=dataset_test[i])
                y_pred, n_clusters, y_pred_test, n_clusters_test = run.cluster_data(encoded_data=encoded_data,
                                                                                    encoded_data_test=encoded_data_test)
                train_acc, test_acc = run.evaluate_spike_sorting(y_pred, y_pred_test, y_true, y_true_test)
                train_acc_lst.append(train_acc)
                test_acc_lst.append(test_acc)
            print("Train Accuracies: ", train_acc_lst)
            print("Test Accuracies: ", test_acc_lst)
            print("Mean Train Accuracy: ", statistics.mean(train_acc_lst), ", Standarddeviation: ",
                  statistics.stdev(train_acc_lst))
            print("Mean Test Accuracy: ", statistics.mean(test_acc_lst), ", Standarddeviation: ",
                  statistics.stdev(test_acc_lst))
            end_time = time.time()
            print("Time Run Execution: ", end_time - start_time)

        else:
            start_time = time.time()
            dataset, dataset_test = run.prepare_data()
            model = run.initialize_model()
            loss_lst, test_loss_lst = run.train(model=model, dataset=dataset, dataset_test=dataset_test)
            encoded_data, encoded_data_test, y_true, y_true_test = run.predict(model=model, dataset=dataset,
                                                                               dataset_test=dataset_test)
            y_pred, n_clusters, y_pred_test, n_clusters_test = run.cluster_data(encoded_data=encoded_data,
                                                                                encoded_data_test=encoded_data_test)

            train_acc, test_acc = run.evaluate_spike_sorting(y_pred, y_pred_test, y_true, y_true_test)
            print("Train Accuracy: ", train_acc)
            print("Test Accuracy: ", test_acc)
            end_time = time.time()
            print("Time Run Execution: ", end_time - start_time)


if args.Model == "PerceiverIO":
    config = Config_PerceiverIO(data_path=args.PathData)
elif args.Model == "AttnAE_1":
    config = Config_AttnAE_1(data_path=args.PathData)
elif args.Model == "AttnAE_2":
    config = Config_AttnAE_2(data_path=args.PathData)
else:
    raise ValueError("please choose a valid Model Type. See Documentation!")

if args.Benchmark:
    run = Run(config=config, benchmark=True)
else:
    run = Run(config=config, benchmark=False)

run.execute_run()
