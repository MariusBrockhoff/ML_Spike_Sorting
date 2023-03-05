# -*- coding: utf-8 -*-
class Config_AutoPerceiver(object):

    # General  

    SAVE_DIR = '/content/drive/My Drive/Data ML/Model data .h5/2022_03_15_Peceiver_Testing/'

    LOAD = False #'run20211212T2105/'
    
    

    # TRAINING HYPERPARAMETERS

    LEARNING_RATE = 1e-5 #1e-4 #1e-6
    
    WITH_WARMUP = False

    LR_WARMUP =  10 #2 #10

    LR_FINAL = 1e-9  #1e-6 1e-8
    
    WITH_WD = False

    WEIGHT_DECAY = 1e-2 #1e-5 1e-7

    WD_FINAL = 1e-4    #1e-41e-6

    NUM_EPOCHS = 50

    BATCH_SIZE = 512
    


    #Architecture

    EMBEDDING_DIM = 64

    SEQ_LEN = 63



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
