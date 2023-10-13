# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import time

#print start time of code execution
print("Start Time Code Exec: ", time.asctime(time.localtime(time.time())))

from utils.run_class import  *
from config_files.config_file_PerceiverIO import *
from config_files.config_AttnE_DenseD import *
from config_files.config_FullTransformer import *
from config_files.config_file_DenseAutoencoder import *


parser = argparse.ArgumentParser()
parser.add_argument('--Pretrain_Method', type=str)
parser.add_argument('--Finetune_Method', type=str)
parser.add_argument('--Model', type=str, required=True)
parser.add_argument('--PathData', type=str, required=True)
parser.add_argument('--Benchmark', action='store_true')
args = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print("devices:", tf.config.list_physical_devices(device_type=None))



if args.Model == "PerceiverIO":
    config = Config_PerceiverIO(data_path=args.PathData)

elif args.Model == "DenseAutoencoder":
    config = Config_DenseAutoencoder(data_path=args.PathData)

elif args.Model == "AttnE_DenseD":
    config = Config_AttnAE(data_path=args.PathData)
    config.MODEL_TYPE = "AttnE_DenseD"

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

if args.Pretrain_Method != "None" and args.Pretrain_Method != None:
    run.execute_pretrain()
if args.Finetune_Method != "None"and args.Finetune_Method != None:
    run.execute_finetune()