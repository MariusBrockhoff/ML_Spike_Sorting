# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import argparse
import statistics
import tensorflow as tf
import time
import wandb

#print start time of code execution
print("Start Time Code Exec: ", time.asctime(time.localtime(time.time())))


from utils.model_initializer import *
from utils.data_preparation import *
from utils.train_models import *
from utils.model_predict import *
from utils.clustering import *
from utils.evaluation import *
from utils.wandb_initializer import *
from utils.finetune_models import *

from config_files.config_file_PerceiverIO import *
from config_files.config_file_DenseAutoencoder import *
from config_files.config_AttnAE import *
from config_files.config_FullTransformer import *
from config_files.config_file_DenseAutoencoder import *
from config_files.config_finetune import *

from models.PerceiverIO import *
from models.DINOPerceiver import *
from models.AttnAE_1 import *
from models.AttnAE_2 import *
from models.FullTransformerAE import *

parser = argparse.ArgumentParser()
parser.add_argument('--Pretrain_Method', type=str, required=True)
parser.add_argument('--Finetune_Method', type=str, required=True)
parser.add_argument('--Model', type=str, required=True)
parser.add_argument('--PathData', type=str, required=True)
parser.add_argument('--Benchmark', action='store_true')
args = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print("devices:", tf.config.list_physical_devices(device_type=None))


class Run:
    """Class for running full Spike sorting pipeline"""

    def __init__(self, config, benchmark, pretrain_method, fine_tune_method):
        self.config = config
        self.benchmark = benchmark
        self.pretrain_method = pretrain_method
        self.fine_tune_method = fine_tune_method

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
        model = model_initializer(self.config)
        return model

    def initialize_wandb(self):
        wandb_initializer(self.config)


    def pretrain(self, model, dataset, dataset_test):
        print('---' * 30)
        print('TRAINING MODEL...')

        loss_lst, test_loss_lst, final_epoch = pretrain_model(model=model, config=self.config, pretrain_method=self.pretrain_method,
                                                                  dataset=dataset, dataset_test=dataset_test,
                                                                  save_weights=self.config.SAVE_WEIGHTS, save_dir=self.config.SAVE_DIR)

        # get number of trainable parameters of model
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams

        print('-' * 20)
        print('trainable parameters:', trainableParams)
        print('non-trainable parameters', nonTrainableParams)
        print('total parameters', totalParams)
        print('-' * 20)

        return loss_lst, test_loss_lst, final_epoch


    def finetune(self, model, dataset, dataset_test):
        print('---' * 30)
        print('FINETUNING MODEL...')
        #load finetune config file
        fintune_config = Config_Finetuning()
        y_finetuned = finetune_model(model=model, config=self.config, finetune_config=fintune_config, finetune_method=self.fine_tune_method,
                                                                  dataset=dataset, dataset_test=dataset_test,
                                                                  load_dir=self.config.SAVE_DIR)

        return y_finetuned

    def predict(self, model, dataset, dataset_test):

        encoded_data, encoded_data_test, y_true, y_true_test = model_predict_latents(config=self.config, model=model, dataset=dataset,
                                                                                     dataset_test=dataset_test)
        return encoded_data, encoded_data_test, y_true, y_true_test

    def cluster_data(self, encoded_data, encoded_data_test):
        print('---' * 30)
        print('CLUSTERING...')
        #if self.config.MODEL_TYPE == "DINO":
            #y_pred, n_clusters = DINO_clustering(data=encoded_data, method=self.config.CLUSTERING_METHOD,
                                            #n_clusters=self.config.N_CLUSTERS,
                                            #eps=self.config.EPS, min_cluster_size=self.config.MIN_CLUSTER_SIZE,
                                            #knn=self.config.KNN)
            #y_pred_test, n_clusters_test = DINO_clustering(data=encoded_data_test, method=self.config.CLUSTERING_METHOD,
                                                      #n_clusters=self.config.N_CLUSTERS, eps=self.config.EPS,
                                                      #min_cluster_size=self.config.MIN_CLUSTER_SIZE,
                                                      #knn=self.config.KNN)
        y_pred, n_clusters = clustering(data=encoded_data, method=self.config.CLUSTERING_METHOD,
                                        n_clusters=self.config.N_CLUSTERS,
                                        eps=self.config.EPS, min_cluster_size=self.config.MIN_CLUSTER_SIZE, knn=self.config.KNN)
        y_pred_test, n_clusters_test = clustering(data=encoded_data_test, method=self.config.CLUSTERING_METHOD,
                                                  n_clusters=self.config.N_CLUSTERS, eps=self.config.EPS,
                                                  min_cluster_size=self.config.MIN_CLUSTER_SIZE, knn=self.config.KNN)
                                                      
        return y_pred, n_clusters, y_pred_test, n_clusters_test

    def evaluate_spike_sorting(self, y_pred, y_true, y_pred_test=None, y_true_test=None):
        print('---' * 30)
        print('EVALUATE RESULTS...')
        if self.config.MODEL_TYPE == "DINO":
            train_acc, test_acc = DINO_evaluate_clustering(y_pred, y_true, y_pred_test, y_true_test)
        else:
            train_acc, test_acc = evaluate_clustering(y_pred, y_true, y_pred_test, y_true_test)
        return train_acc, test_acc

    #TODO: pretrain workflow: prepare data --> initialize model --> choose training --> train --> evaluate --> save
    def execute_pretrain(self): #TODO: add train_method as argument (such as DINO, NNCLR, etc.) in addition to pure reconstruction/unsupervised
        if self.benchmark:
            dataset, dataset_test = run.prepare_data()
            train_acc_lst = []
            test_acc_lst = []
            for i in range(len(dataset)):
                run.initialize_wandb()
                start_time = time.time()
                model = run.initialize_model()
                loss_lst, test_loss_lst, final_epoch = run.pretrain(model=model, dataset=dataset[i], dataset_test=dataset_test[i])
                encoded_data, encoded_data_test, y_true, y_true_test = run.predict(model=model, dataset=dataset[i],
                                                                                   dataset_test=dataset_test[i])
                y_pred, n_clusters, y_pred_test, n_clusters_test = run.cluster_data(encoded_data=encoded_data,
                                                                                    encoded_data_test=encoded_data_test)
                train_acc, test_acc = run.evaluate_spike_sorting(y_pred, y_true, y_pred_test, y_true_test)
                train_acc_lst.append(train_acc)
                test_acc_lst.append(test_acc)
                end_time = time.time()
                print("Time Run Execution: ", end_time - start_time)
                print("Train Acc: ", train_acc)
                print("Test Acc: ", test_acc)
            print("Train Accuracies: ", train_acc_lst)
            print("Test Accuracies: ", test_acc_lst)
            print("Mean Train Accuracy: ", statistics.mean(train_acc_lst), ", Standarddeviation: ",
                  statistics.stdev(train_acc_lst))
            print("Mean Test Accuracy: ", statistics.mean(test_acc_lst), ", Standarddeviation: ",
                  statistics.stdev(test_acc_lst))

        else:
            run.initialize_wandb()
            start_time = time.time()
            dataset, dataset_test = run.prepare_data()
            model = run.initialize_model()
            loss_lst, test_loss_lst, final_epoch = run.pretrain(model=model, dataset=dataset, dataset_test=dataset_test)
            encoded_data, encoded_data_test, y_true, y_true_test = run.predict(model=model, dataset=dataset,
                                                                               dataset_test=dataset_test)
            y_pred, n_clusters, y_pred_test, n_clusters_test = run.cluster_data(encoded_data=encoded_data,
                                                                                encoded_data_test=encoded_data_test)

            train_acc, test_acc = run.evaluate_spike_sorting(y_pred, y_true, y_pred_test, y_true_test)
            print("Train Accuracy: ", train_acc)
            print("Test Accuracy: ", test_acc)
            end_time = time.time()
            print("Time Run Execution: ", end_time - start_time)

    def execute_finetune(self): #TODO: add finetuning workflow: prepare data --> load model --> choose + initialize finetuning --> fine-tune --> evaluate --> save
        start_time = time.time()
        dataset, dataset_test = run.prepare_data()
        model = run.initialize_model()
        y_pred_finetuned, y_true = run.finetune(model=model, dataset=dataset, dataset_test=dataset_test)
        train_acc, _ = run.evaluate_spike_sorting(y_pred_finetuned, y_true)
        print("Accuracy after Finetuning: ", train_acc)
        end_time = time.time()
        print("Time Finetuning Execution: ", end_time - start_time)


if args.Model == "PerceiverIO":
    config = Config_PerceiverIO(data_path=args.PathData)

elif args.Model == "DenseAutoencoder":
    config = Config_DenseAutoencoder(data_path=args.PathData)

elif args.Model == "AttnAE_1":
    config = Config_AttnAE(data_path=args.PathData)
    config.MODEL_TYPE = "AttnAE_1"
    assert config.D_MODEL % config.NUM_ATTN_HEADS == 0
    
elif args.Model == "AttnAE_2":
    config = Config_AttnAE(data_path=args.PathData)
    config.MODEL_TYPE = "AttnAE_2"
    assert config.D_MODEL % config.NUM_ATTN_HEADS == 0

elif args.Model == "AE":
    config = Config_AttnAE(data_path=args.PathData)
    config.MODEL_TYPE = "AE"

elif args.Model == "FullTransformer":
    config = Config_FullTransformer(data_path=args.PathData)
    
else:
    raise ValueError("please choose a valid Model Type. See Documentation!")

if args.Benchmark:
    run = Run(config=config, benchmark=True, pretrain_method=args.Pretrain_Method, fine_tune_method=args.Finetune_Method)
    
else:
    run = Run(config=config, benchmark=False, pretrain_method=args.Pretrain_Method, fine_tune_method=args.Finetune_Method)

if args.Pretrain_Method != "None":
    run.execute_pretrain()
if args.Finetune_Method != "None":
    run.execute_finetune()