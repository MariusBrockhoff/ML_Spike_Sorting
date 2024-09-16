import argparse
import numpy as np

from utils.data_preparation import data_preparation
from config_files.config_data_preprocessing import Config_Preprocessing
from config_files.config_pretraining import Config_Pretraining
from config_files.config_finetune import Config_Finetuning
from config_files.config_file_DenseAutoencoder import Config_DenseAutoencoder

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--PathData', type=str, required=True)
args = parser.parse_args()

model_config = Config_DenseAutoencoder()

data_preprocessing_config = Config_Preprocessing(args.PathData)
pretraining_config = Config_Pretraining(args.PathData, model_config.MODEL_TYPE)
fintune_config = Config_Finetuning(args.PathData, model_config.MODEL_TYPE)

dataset, dataset_test, pretraining_config, fintune_config = data_preparation(model_config=model_config,
                                                                             data_preprocessing_config=data_preprocessing_config,
                                                                             pretraining_config=pretraining_config,
                                                                             fintune_config=fintune_config,
                                                                             benchmark=False)

x_train = np.concatenate([x for x, y in dataset], axis=0)
y_train = np.concatenate([y for x, y in dataset], axis=0)
x_test = np.concatenate([x for x, y in dataset_test], axis=0)
y_test = np.concatenate([y for x, y in dataset_test], axis=0)

print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)
print("x_test.shape: ", x_test.shape)
print("y_test.shape: ", y_test.shape)

