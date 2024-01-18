import argparse

from utils.run_class import *
from config_files.config_file_DenseAutoencoder import *

# print start time of code execution
print("Start Time Code Exec: ", time.asctime(time.localtime(time.time())))

parser = argparse.ArgumentParser()
parser.add_argument('--Pretrain_Method', type=str)
parser.add_argument('--Finetune_Method', type=str)
parser.add_argument('--Model', type=str, required=True)
parser.add_argument('--PathData', type=str, required=True)
args = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print("devices:", tf.config.list_physical_devices(device_type=None))


if args.Model == "DenseAutoencoder":
    model_config = Config_DenseAutoencoder()
    
else:
    raise ValueError("please choose a valid Model Type. See Documentation!")

run = Run(model_config=model_config,
          data_path=args.PathData,
          pretrain_method=args.Pretrain_Method,
          fine_tune_method=args.Finetune_Method)

if args.Pretrain_Method != "None" and args.Pretrain_Method is not None:
    run.execute_pretrain()
if args.Finetune_Method != "None" and args.Finetune_Method is not None:
    run.execute_finetune()
