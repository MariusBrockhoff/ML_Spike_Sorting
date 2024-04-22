import argparse
from utils.run_class import *
from config_files.config_file_DenseAutoencoder import *
import tensorflow as tf

# Display the start time of the code execution
print("Start Time Code Exec: ", time.asctime(time.localtime(time.time())))

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--Pretrain_Method', type=str)
parser.add_argument('--Finetune_Method', type=str)
parser.add_argument('--Model', type=str, required=True)
parser.add_argument('--PathData', type=str, required=True)
parser.add_argument('--Benchmark', action='store_true')
args = parser.parse_args()

# Display GPU information
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print("devices:", tf.config.list_physical_devices(device_type=None))

# Initialize model configuration based on the provided model type
if args.Model == "DenseAutoencoder":
    model_config = Config_DenseAutoencoder()

else:
    raise ValueError("please choose a valid Model Type. See Documentation!")

# Initialize the Run class based on provided arguments
if args.Benchmark:
    run = Run(model_config=model_config,
              data_path=args.PathData,
              benchmark=True,
              pretrain_method=args.Pretrain_Method,
              fine_tune_method=args.Finetune_Method)

else:
    run = Run(model_config=model_config,
              data_path=args.PathData,
              benchmark=False,
              pretrain_method=args.Pretrain_Method,
              fine_tune_method=args.Finetune_Method)

# Execute pretraining and finetuning if specified
if args.Pretrain_Method != "None" and args.Pretrain_Method is not None:
    run.execute_pretrain()
if args.Finetune_Method != "None" and args.Finetune_Method is not None:
    run.execute_finetune()
