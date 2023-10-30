import statistics
import time

from utils.model_initializer import *
from utils.data_preparation import *
from utils.pretrain_models import *
from utils.model_predict import *
from utils.clustering import *
from utils.evaluation import *
from utils.wandb_initializer import *
from utils.finetune_models import *

from config_files.config_finetune import *
from config_files.config_pretraining import *





class Run:
    """Class for running full Spike sorting pipeline"""

    def __init__(self, model_config, data_path, benchmark, pretrain_method, fine_tune_method):
        self.model_config = model_config
        self.data_path = data_path
        self.benchmark = benchmark
        self.pretrain_method = pretrain_method
        self.fine_tune_method = fine_tune_method
        self.pretraining_config = Config_Pretraining(self.data_path, self.model_config.MODEL_TYPE)
        self.fintune_config = Config_Finetuning(self.data_path, self.model_config.MODEL_TYPE)

    """def extract_spikes(self, self.raw_file_name, self.data_path, self.is_big_file, self.filter_frequencies, self.filtering_method, self.order, self.save_path, self.min_TH, self.dat_points_pre_min, self.dat_points_post_min, self.max_TH, self.chunck_len, self.refrec_period, self.reject_channels, self.file_name, self.save):
        recording_data, electrode_stream, fsample = file_opener_raw_recording_data(self.raw_file_name, self.data_path, self.is_big_file=False)
        filtered_signal = filter_signal(recording_data, self.filter_frequencies, fsample, self.filtering_method="Butter_bandpass", self.order=2)
        spike_file = spike_detection(filtered_signal, electrode_stream, fsample, self.save_path, self.min_TH, self.dat_points_pre_min=20, self.dat_points_post_min=44, self.max_TH=-30, self.chunck_len=300, self.refrec_period=0.002, self.reject_channels=[None], self.file_name=None, self.save=True):
    """

    def prepare_data(self):
        print('---' * 30)
        print('PREPARING DATA...')
        dataset, dataset_test, self.pretraining_config, self.fintune_config = data_preparation(self.model_config, self.pretraining_config, self.pretrain_method, self.fintune_config, self.benchmark)
        return dataset, dataset_test

    def initialize_model(self):
        model = model_initializer(self.model_config)
        return model

    def initialize_wandb(self, method):
        wandb_initializer(self.model_config, self.pretraining_config, self.fintune_config, method)

    def pretrain(self, model, dataset, dataset_test):
        print('---' * 30)
        print('PRETRAINING MODEL...')

        save_pseudo = pretrain_model(model=model, model_config=self.model_config,
                      pretraining_config=self.pretraining_config,
                      pretrain_method=self.pretrain_method,
                      dataset=dataset, dataset_test=dataset_test,
                      save_weights=self.pretraining_config.SAVE_WEIGHTS,
                      save_dir=self.pretraining_config.SAVE_DIR)
        return save_pseudo

    def finetune(self, model, dataset, dataset_test):
        print('---' * 30)
        print('FINETUNING MODEL...')
        y_finetuned = finetune_model(model=model, finetune_config=self.fintune_config,
                                     finetune_method=self.fine_tune_method,
                                     dataset=dataset, dataset_test=dataset_test)

        return y_finetuned

    def predict(self, model, dataset, dataset_test):

        encoded_data, encoded_data_test, y_true, y_true_test = model_predict_latents(model=model,
                                                                                     dataset=dataset,
                                                                                     dataset_test=dataset_test)
        return encoded_data, encoded_data_test, y_true, y_true_test

    def cluster_data(self, encoded_data, encoded_data_test):
        print('---' * 30)
        print('CLUSTERING...')
        y_pred, n_clusters = clustering(data=encoded_data, method=self.pretraining_config.CLUSTERING_METHOD,
                                        n_clusters=self.pretraining_config.N_CLUSTERS,
                                        eps=self.pretraining_config.EPS, min_cluster_size=self.pretraining_config.MIN_CLUSTER_SIZE,
                                        knn=self.pretraining_config.KNN)
        y_pred_test, n_clusters_test = clustering(data=encoded_data_test, method=self.pretraining_config.CLUSTERING_METHOD,
                                                  n_clusters=self.pretraining_config.N_CLUSTERS, eps=self.pretraining_config.EPS,
                                                  min_cluster_size=self.MIN_CLUSTER_SIZE, knn=self.pretraining_config.KNN)

        return y_pred, n_clusters, y_pred_test, n_clusters_test

    def evaluate_spike_sorting(self, y_pred, y_true, y_pred_test=None, y_true_test=None):
        print('---' * 30)
        print('EVALUATE RESULTS...')
        train_acc, test_acc = evaluate_clustering(y_pred, y_true, y_pred_test, y_true_test)
        return train_acc, test_acc



    def execute_pretrain(self):
        if self.benchmark:
            dataset, dataset_test = self.prepare_data()
            train_acc_lst = []
            test_acc_lst = []
            for i in range(len(dataset)):
                start_time = time.time()
                self.initialize_wandb(self.pretrain_method)
                model = self.initialize_model()
                self.pretrain(model=model, dataset=dataset[i], dataset_test=dataset_test[i])
                if self.pretrain_method == "reconstruction":
                    encoded_data, encoded_data_test, y_true, y_true_test = self.predict(model=model, dataset=dataset[i],
                                                                                       dataset_test=dataset_test[i])
                    y_pred, n_clusters, y_pred_test, n_clusters_test = self.cluster_data(encoded_data=encoded_data,
                                                                                        encoded_data_test=encoded_data_test)
                    train_acc, test_acc = self.evaluate_spike_sorting(y_pred, y_true, y_pred_test, y_true_test)
                    train_acc_lst.append(train_acc)
                    test_acc_lst.append(test_acc)
                    print("Train Acc: ", train_acc)
                    print("Test Acc: ", test_acc)
                end_time = time.time()
                print("Time Run Execution: ", end_time - start_time)

        else:
            start_time = time.time()
            self.initialize_wandb(self.pretrain_method)
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            self.fintune_config.PRETRAINED_SAVE_DIR = self.pretrain(model=model, dataset=dataset, dataset_test=dataset_test)
            if self.pretrain_method == "reconstruction":
                encoded_data, encoded_data_test, y_true, y_true_test = self.predict(model=model, dataset=dataset,
                                                                                   dataset_test=dataset_test)
                y_pred, n_clusters, y_pred_test, n_clusters_test = self.cluster_data(encoded_data=encoded_data,
                                                                                    encoded_data_test=encoded_data_test)

                train_acc, test_acc = self.evaluate_spike_sorting(y_pred, y_true, y_pred_test, y_true_test)
                print("Train Accuracy: ", train_acc)
                print("Test Accuracy: ", test_acc)
            end_time = time.time()
            print("Time Run Execution: ", end_time - start_time)

    def execute_finetune(self):
        start_time = time.time()
        self.initialize_wandb(self.fine_tune_method)
        dataset, dataset_test = self.prepare_data()
        model = self.initialize_model()
        y_pred_finetuned, y_true = self.finetune(model=model, dataset=dataset, dataset_test=dataset_test)
        train_acc, _ = self.evaluate_spike_sorting(y_pred_finetuned, y_true)
        print("Accuracy after Finetuning: ", train_acc)
        end_time = time.time()
        print("Time Finetuning Execution: ", end_time - start_time)