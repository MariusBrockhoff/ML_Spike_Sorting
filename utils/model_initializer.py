# -*- coding: utf-8 -*-
"""
Load Spike data and prepare for training of model
"""
import numpy as np
import os
import sys

#cwd = os.getcwd()
#root_folder = os.sep+"ML_Spike_Sorting"
#sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep)

from models.PerceiverIO import *
from models.DINOPerceiver import *
from models.AttnE_DenseD import *
from models.DenseAutoencoder import *
from models.FullTransformerAE import *

def model_initializer(config):
    print('---' * 30)
    print('INITIALIZING MODEL...')
    if config.MODEL_TYPE == "PerceiverIO":
        model = AutoPerceiver(embedding_dim=config.EMBEDDING_DIM,
                              seq_len=config.SEQ_LEN,
                              latent_len=config.LATENT_LEN,
                              ENC_number_of_layers=config.ENC_NUMBER_OF_LAYERS,
                              ENC_state_index=config.ENC_STATE_INDEX,
                              ENC_state_channels=config.ENC_STATE_CHANNELS,
                              ENC_dff=config.ENC_DFF,
                              ENC_x_attn_dim=config.ENC_X_ATTN_DIM,
                              ENC_x_attn_heads=config.ENC_X_ATTN_HEADS,
                              ENC_depth=config.ENC_DEPTH,
                              ENC_attn_dim=config.ENC_SELF_ATTN_DIM,
                              ENC_attn_heads=config.ENC_NUM_ATTN_HEADS,
                              ENC_dropout_rate=config.ENC_DROPOUT_RATE,
                              DEC_number_of_layers=config.DEC_NUMBER_OF_LAYERS,
                              DEC_state_index=config.DEC_STATE_INDEX,
                              DEC_state_channels=config.DEC_STATE_CHANNELS,
                              DEC_dff=config.DEC_DFF,
                              DEC_x_attn_dim=config.DEC_X_ATTN_DIM,
                              DEC_x_attn_heads=config.DEC_X_ATTN_HEADS,
                              DEC_depth=config.DEC_DEPTH,
                              DEC_attn_dim=config.DEC_SELF_ATTN_DIM,
                              DEC_attn_heads=config.DEC_NUM_ATTN_HEADS,
                              DEC_dropout_rate=config.DEC_DROPOUT_RATE)


    elif config.MODEL_TYPE == "AttnE_DenseD":
        model = Attention_AE(embedding_dim=config.EMBEDDING_DIM,
                                dff=config.DFF,
                                seq_len=config.SEQ_LEN,
                                latent_len=config.LATENT_LEN,
                                enc_depth=config.ENC_DEPTH,
                                enc_attn_dim=int(config.EMBEDDING_DIM / config.ENC_NUM_ATTN_HEADS),
                                enc_attn_heads=config.ENC_NUM_ATTN_HEADS,
                                enc_dropout_rate=config.ENC_DROPOUT_RATE,
                                dec_layers=config.DEC_LAYERS)

    elif config.MODEL_TYPE == "FullTransformer":
        model = FullTransformer(embedding_dim=config.EMBEDDING_DIM,
                                dff=config.DFF,
                                seq_len=config.SEQ_LEN,
                                latent_len=config.LATENT_LEN,
                                enc_depth=config.ENC_DEPTH,
                                enc_attn_dim=int(config.EMBEDDING_DIM / config.ENC_NUM_ATTN_HEADS),
                                enc_attn_heads=config.ENC_NUM_ATTN_HEADS,
                                enc_dropout_rate=config.ENC_DROPOUT_RATE,
                                dec_depth=config.DEC_DEPTH,
                                dec_attn_dim=int(config.EMBEDDING_DIM / config.DEC_NUM_ATTN_HEADS),
                                dec_attn_heads=config.DEC_NUM_ATTN_HEADS,
                                dec_dropout_rate=config.DEC_DROPOUT_RATE)



    elif config.MODEL_TYPE == "DenseAutoencoder":
        model = DenseAutoencoder(dims=config.DIMS,
                            act=config.ACT)


    elif config.MODEL_TYPE == "DINO":  # TODO: define DINO as a pre-train class and make it compatible to any pre-defined model
        m_student = Perceiver(Embedding_dim=config.EMBEDDING_DIM,
                              seq_len=config.CROP_PCTS[1],
                              number_of_layers=config.NUMBER_OF_LAYERS,
                              latent_len=config.LATENT_LEN,
                              state_index=config.STATE_INDEX,
                              state_channels=config.STATE_CHANNELS,
                              dff=config.DFF,
                              x_attn_dim=config.X_ATTN_DIM,
                              x_attn_heads=config.X_ATTN_HEADS,
                              depth=config.DEPTH,
                              attn_dim=config.SELF_ATTN_DIM,
                              attn_heads=config.NUM_ATTN_HEADS,
                              dropout_rate=config.DROPOUT_RATE)

        m_teacher = Perceiver(Embedding_dim=config.EMBEDDING_DIM,
                              seq_len=config.CROP_PCTS[0],
                              number_of_layers=config.NUMBER_OF_LAYERS,
                              latent_len=config.LATENT_LEN,
                              state_index=config.STATE_INDEX,
                              state_channels=config.STATE_CHANNELS,
                              dff=config.DFF,
                              x_attn_dim=config.X_ATTN_DIM,
                              x_attn_heads=config.X_ATTN_HEADS,
                              depth=config.DEPTH,
                              attn_dim=config.SELF_ATTN_DIM,
                              attn_heads=config.NUM_ATTN_HEADS,
                              dropout_rate=config.DROPOUT_RATE)

        model = [m_student, m_teacher]

    return model


