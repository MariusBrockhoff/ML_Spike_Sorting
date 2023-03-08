# -*- coding: utf-8 -*-
class Config_AutoPerceiver(object):

    # Data

    DATA_SAVE_PATH = "C:\\Users\\marib\\Documents\\Github\\ML_Spike_Sorting\\Data\\Small_SpikesFile_1.pkl"

    DATA_PREP_METHOD = "gradient"

    DATA_NORMALIZATION = "Standard"

    TRAIN_TEST_SPLIT = 0.1
    
    

    # TRAINING HYPERPARAMETERS

    LEARNING_RATE = 1e-5 #1e-4 #1e-6
    
    WITH_WARMUP = False

    LR_WARMUP =  10 #2 #10

    LR_FINAL = 1e-9  #1e-6 1e-8
    
    WITH_WD = False

    WEIGHT_DECAY = 1e-2 #1e-5 1e-7

    WD_FINAL = 1e-4    #1e-41e-6

    NUM_EPOCHS = 1

    BATCH_SIZE = 512

    # General

    SAVE_WEIGHTS = False

    SAVE_DIR = None

    LOAD = False

    LOAD_DIR = None
    

    #Architecture

    MODEL_TYPE = "AutoPerceiver"

    EMBEDDING_DIM = 64

    if DATA_PREP_METHOD == "gradient":
        SEQ_LEN = 63
    else:
        SEQ_LEN = 64



    # Encoder 

    ENC_NUMBER_OF_LAYERS = 3 

    ENC_LATENT_LEN = 512

    ENC_STATE_INDEX = 32 

    ENC_STATE_CHANNELS = 512 

    ENC_DFF = ENC_STATE_CHANNELS*2

    ENC_X_ATTN_HEADS = 8
    
    ENC_X_ATTN_DIM = int(ENC_STATE_CHANNELS / ENC_X_ATTN_HEADS)

    ENC_DEPTH = 4
 
    ENC_NUM_ATTN_HEADS = 8

    ENC_SELF_ATTN_DIM = int(ENC_STATE_CHANNELS / ENC_NUM_ATTN_HEADS)

    ENC_DROPOUT_RATE = 0



    #Decoder

    DEC_NUMBER_OF_LAYERS = 1

    DEC_STATE_INDEX = 32 

    DEC_STATE_CHANNELS = 512 

    DEC_DFF = DEC_STATE_CHANNELS*2

    DEC_X_ATTN_HEADS = 8
    
    DEC_X_ATTN_DIM = int(DEC_STATE_CHANNELS / DEC_X_ATTN_HEADS)

    DEC_DEPTH = 4
 
    DEC_NUM_ATTN_HEADS = 8

    DEC_SELF_ATTN_DIM = int(DEC_STATE_CHANNELS / DEC_NUM_ATTN_HEADS)

    DEC_DROPOUT_RATE = 0
    
    
    #Clustering
    
    CLUSTERING_METHOD = "Kmeans"
    
    N_CLUSTERS = 5
    
    EPS = None
    
    MIN_CLUSTER_SIZE = 1000