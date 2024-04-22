import time

from utils.model_initializer import *
from utils.data_preparation import *
from utils.model_predict import *
from utils.clustering import *
from utils.wandb_initializer import *
from utils.finetune_models import *
from utils.file_opener_raw_recording_data import *
from utils.filter_signal import *
from utils.spike_detection import *


from config_files.config_finetune import *
from config_files.config_pretraining import *
from config_files.config_data_preprocessing import *


class Run:
    """
    Class for running the full Spike sorting pipeline on raw MEA recordings (MCS file format) (if benchmark=False)
    or benchmarking the method on simulated spike files (if benchmark=True)

    This class integrates different components of the spike sorting pipeline, including
    data preparation, model initialization, pretraining, finetuning, prediction, clustering,
    and evaluation.

    Attributes:
        model_config: Configuration for the model.
        data_path: Path to the data.
        pretrain_method: Method for pretraining the model.
        fine_tune_method: Method for finetuning the model.
        pretraining_config: Configuration for pretraining.
        finetune_config: Configuration for finetuning.

    Methods:
        prepare_data(): Prepare the data for training and testing.
        initialize_model(): Initialize the machine learning model.
        initialize_wandb(): Initialize Weights & Biases for tracking experiments.
        pretrain(): Pretrain the model.
        finetune(): Finetune the model.
        predict(): Predict latent representations using the model.
        cluster_data(): Perform clustering on the encoded data.
        evaluate_spike_sorting(): Evaluate the results of spike sorting.
        execute_pretrain(): Execute the pretraining phase.
        execute_finetune(): Execute the finetuning phase.
    """

    def __init__(self, model_config, data_path, benchmark, pretrain_method, fine_tune_method):
        self.model_config = model_config
        self.data_path = data_path
        self.benchmark = benchmark
        self.pretrain_method = pretrain_method
        self.fine_tune_method = fine_tune_method
        self.data_preprocessing_config = Config_Preprocessing(self.data_path)
        self.pretraining_config = Config_Pretraining(self.data_path, self.model_config.MODEL_TYPE)
        self.finetune_config = Config_Finetuning(self.data_path, self.model_config.MODEL_TYPE)


    def extract_spikes_from_raw_recording(self):
        recording_data, electrode_stream, fsample = file_opener_raw_recording_data(self.data_preprocessing_config)
        filtered_signal = filter_signal(recording_data, fsample, self.data_preprocessing_config)
        spike_file = spike_detection(filtered_signal, electrode_stream, fsample, self.data_preprocessing_config)

    def prepare_data(self):
        print('---' * 30)
        print('PREPARING DATA...')
        dataset, dataset_test, self.pretraining_config, self.finetune_config = data_preparation(self.model_config,
                                                                                                self.data_preprocessing_config,
                                                                                                self.pretraining_config,
                                                                                                self.finetune_config,
                                                                                                self.benchmark)
        return dataset, dataset_test

    def initialize_model(self):
        model = model_initializer(self.model_config)
        return model

    def initialize_wandb(self, method):
        wandb_initializer(self.model_config, self.pretraining_config, self.finetune_config, method)

    def pretrain(self, model, dataset, dataset_test):
        print('---' * 30)
        print('PRETRAINING MODEL...')

        pretrain_model(model=model,
                       pretraining_config=self.pretraining_config,
                       pretrain_method=self.pretrain_method,
                       dataset=dataset,
                       dataset_test=dataset_test,
                       save_weights=self.pretraining_config.SAVE_WEIGHTS,
                       save_dir=self.pretraining_config.SAVE_DIR)

    def finetune(self, model, dataset, dataset_test):
        print('---' * 30)
        print('FINETUNING MODEL...')
        y_finetuned, y_true = finetune_model(model=model, finetune_config=self.finetune_config,
                                     finetune_method=self.fine_tune_method,
                                     dataset=dataset, dataset_test=dataset_test, benchmark=self.benchmark)

        return y_finetuned, y_true

    def predict(self, model, dataset, dataset_test):
        encoded_data, encoded_data_test, y_true, y_true_test = model_predict_latents(model=model,
                                                                                     dataset=dataset,
                                                                                     dataset_test=dataset_test)

        return encoded_data, encoded_data_test, y_true, y_true_test

    def cluster_data(self, encoded_data, encoded_data_test):
        print('---' * 30)
        print('CLUSTERING...')
        y_pred, n_clusters = clustering(data=encoded_data,
                                        method=self.pretraining_config.CLUSTERING_METHOD,
                                        n_clusters=self.pretraining_config.N_CLUSTERS,
                                        eps=self.pretraining_config.EPS,
                                        min_cluster_size=self.pretraining_config.MIN_CLUSTER_SIZE,
                                        knn=self.pretraining_config.KNN)
        y_pred_test, n_clusters_test = clustering(data=encoded_data_test,
                                                  method=self.pretraining_config.CLUSTERING_METHOD,
                                                  n_clusters=self.pretraining_config.N_CLUSTERS,
                                                  eps=self.pretraining_config.EPS,
                                                  min_cluster_size=self.pretraining_config.MIN_CLUSTER_SIZE,
                                                  knn=self.pretraining_config.KNN)

        return y_pred, n_clusters, y_pred_test, n_clusters_test

    def evaluate_spike_sorting(self, y_pred, y_true, y_pred_test=None, y_true_test=None):
        print('---' * 30)
        print('EVALUATE RESULTS...')
        train_acc, test_acc = evaluate_clustering(y_pred, y_true, y_pred_test, y_true_test)
        return train_acc, test_acc

    def execute_pretrain(self):
        if self.benchmark:
            start_time = time.time()
            self.initialize_wandb(self.pretrain_method)
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            self.pretrain(model=model, dataset=dataset, dataset_test=dataset_test)
            encoded_data, encoded_data_test, y_true, y_true_test = self.predict(model=model,
                                                                                dataset=dataset,
                                                                                dataset_test=dataset_test)
            y_pred, n_clusters, y_pred_test, n_clusters_test = self.cluster_data(encoded_data=encoded_data,
                                                                                 encoded_data_test=encoded_data_test)

            train_acc, test_acc = self.evaluate_spike_sorting(y_pred, y_true, y_pred_test, y_true_test)
            print("Train Accuracy: ", train_acc)
            print("Test Accuracy: ", test_acc)
            end_time = time.time()
            print("Time Run Execution: ", end_time - start_time)
        else:
            start_time = time.time()
            self.initialize_wandb(self.pretrain_method)
            self.extract_spikes_from_raw_recording()
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            self.pretrain(model=model, dataset=dataset, dataset_test=dataset_test)
            encoded_data, encoded_data_test, y_true, y_true_test = self.predict(model=model,
                                                                                dataset=dataset,
                                                                                dataset_test=dataset_test)
            y_pred, n_clusters, y_pred_test, n_clusters_test = self.cluster_data(encoded_data=encoded_data,
                                                                                 encoded_data_test=encoded_data_test)
            end_time = time.time()
            print("Time Run Execution: ", end_time - start_time)

    def execute_finetune(self):
        if self.benchmark:
            start_time = time.time()
            self.initialize_wandb(self.fine_tune_method)
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            y_pred_finetuned, y_true = self.finetune(model=model, dataset=dataset, dataset_test=dataset_test)
            train_acc, _ = self.evaluate_spike_sorting(y_pred_finetuned, y_true)
            print("Accuracy after Finetuning: ", train_acc)
            end_time = time.time()
            print("Time Finetuning Execution: ", end_time - start_time)
        else:
            start_time = time.time()
            self.initialize_wandb(self.fine_tune_method)
            dataset, dataset_test = self.prepare_data()
            model = self.initialize_model()
            y_pred_finetuned, y_true = self.finetune(model=model, dataset=dataset, dataset_test=dataset_test)

            end_time = time.time()
            print("Time Finetuning Execution: ", end_time - start_time)

