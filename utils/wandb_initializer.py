# -*- coding: utf-8 -*-
"""
Load Spike data and prepare for training of model
"""
import wandb

def wandb_initializer(config):
    if config.MODEL_TYPE == "PerceiverIO":
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.MODEL_TYPE,
            # track hyperparameters and run metadata with wandb.config
            config={"Model": config.MODEL_TYPE,
                    "DATA_PREP_METHOD": config.DATA_PREP_METHOD,
                    "DATA_NORMALIZATION": config.DATA_NORMALIZATION,
                    "TRAIN_TEST_SPLIT": config.TRAIN_TEST_SPLIT,
                    "LEARNING_RATE": config.LEARNING_RATE,
                    "WITH_WARMUP": config.WITH_WARMUP,
                    "LR_WARMUP": config.LR_WARMUP,
                    "LR_FINAL": config.LR_FINAL,
                    "NUM_EPOCHS": config.NUM_EPOCHS,
                    "BATCH_SIZE": config.BATCH_SIZE,
                    "EMBEDDING_DIM": config.EMBEDDING_DIM,
                    "SEQ_LEN": config.SEQ_LEN,
                    "LATENT_LEN": config.LATENT_LEN,
                    "ENC_NUMBER_OF_LAYERS": config.ENC_NUMBER_OF_LAYERS,
                    "ENC_STATE_INDEX": config.ENC_STATE_INDEX,
                    "ENC_STATE_CHANNELS": config.ENC_STATE_CHANNELS,
                    "ENC_DFF": config.ENC_DFF,
                    "ENC_X_ATTN_HEADS": config.ENC_X_ATTN_HEADS,
                    "ENC_X_ATTN_DIM": config.ENC_X_ATTN_DIM,
                    "ENC_DEPTH": config.ENC_DEPTH,
                    "ENC_NUM_ATTN_HEADS": config.ENC_NUM_ATTN_HEADS,
                    "ENC_SELF_ATTN_DIM": config.ENC_SELF_ATTN_DIM,
                    "ENC_DROPOUT_RATE": config.ENC_DROPOUT_RATE,
                    "DEC_NUMBER_OF_LAYERS": config.DEC_NUMBER_OF_LAYERS,
                    "DEC_STATE_INDEX": config.DEC_STATE_INDEX,
                    "DEC_STATE_CHANNELS": config.DEC_STATE_CHANNELS,
                    "DEC_DFF": config.DEC_DFF,
                    "DEC_X_ATTN_HEADS": config.DEC_X_ATTN_HEADS,
                    "DEC_X_ATTN_DIM": config.DEC_X_ATTN_DIM,
                    "DEC_DEPTH": config.DEC_DEPTH,
                    "DEC_NUM_ATTN_HEADS": config.DEC_NUM_ATTN_HEADS,
                    "DEC_SELF_ATTN_DIM": config.DEC_SELF_ATTN_DIM,
                    "DEC_DROPOUT_RATE": config.DEC_DROPOUT_RATE,
                    "CLUSTERING_METHOD": config.CLUSTERING_METHOD,
                    "N_CLUSTERS": config.N_CLUSTERS})

    elif config.MODEL_TYPE == "DenseAutoencoder":
        wandb.init(
                # set the wandb project where this run will be logged
                project=config.MODEL_TYPE,
                # track hyperparameters and run metadata with wandb.config
                config={"Model": config.MODEL_TYPE,
                        "DATA_PREP_METHOD": config.DATA_PREP_METHOD,
                        "DATA_NORMALIZATION": config.DATA_NORMALIZATION,
                        "LEARNING_RATE": config.LEARNING_RATE,
                        "WITH_WARMUP": config.WITH_WARMUP,
                        "LR_WARMUP": config.LR_WARMUP,
                        "LR_FINAL": config.LR_FINAL,
                        "NUM_EPOCHS": config.NUM_EPOCHS,
                        "BATCH_SIZE": config.BATCH_SIZE,
                        "DIMS": config.DIMS,
                        "ACT": config.ACT,
                        "DATA_AUG": config.DATA_AUG})

    elif config.MODEL_TYPE == "AttnAE_1":
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.MODEL_TYPE,
            # track hyperparameters and run metadata with wandb.config
            config={"Model": config.MODEL_TYPE,
                    "DATA_PREP_METHOD": config.DATA_PREP_METHOD,
                    "DATA_NORMALIZATION": config.DATA_NORMALIZATION,
                    "LEARNING_RATE": config.LEARNING_RATE,
                    "WITH_WARMUP": config.WITH_WARMUP,
                    "LR_WARMUP": config.LR_WARMUP,
                    "LR_FINAL": config.LR_FINAL,
                    "NUM_EPOCHS": config.NUM_EPOCHS,
                    "BATCH_SIZE": config.BATCH_SIZE,
                    "REG_VALUE": config.REG_VALUE,
                    "DROPOUT_RATE": config.DROPOUT_RATE,
                    "DATA_PREP": config.DATA_PREP,
                    "ENC_DEPTH": config.ENC_DEPTH,
                    "DFF": config.DFF,
                    "NUM_ATTN_HEADS": config.NUM_ATTN_HEADS,
                    "DEC_LAYERS": config.DEC_LAYERS,
                    "D_MODEL": config.D_MODEL,
                    "LATENT_LEN": config.LATENT_LEN,
                    "DATA_AUG": config.DATA_AUG})

    elif config.MODEL_TYPE == "AttnAE_2":
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.MODEL_TYPE,
            # track hyperparameters and run metadata with wandb.config
            config={"Model": config.MODEL_TYPE,
                    "DATA_PREP_METHOD": config.DATA_PREP_METHOD,
                    "DATA_NORMALIZATION": config.DATA_NORMALIZATION,
                    "LEARNING_RATE": config.LEARNING_RATE,
                    "WITH_WARMUP": config.WITH_WARMUP,
                    "LR_WARMUP": config.LR_WARMUP,
                    "LR_FINAL": config.LR_FINAL,
                    "NUM_EPOCHS": config.NUM_EPOCHS,
                    "BATCH_SIZE": config.BATCH_SIZE,
                    "REG_VALUE": config.REG_VALUE,
                    "DROPOUT_RATE": config.DROPOUT_RATE,
                    "DATA_PREP": config.DATA_PREP,
                    "ENC_DEPTH": config.ENC_DEPTH,
                    "DFF": config.DFF,
                    "NUM_ATTN_HEADS": config.NUM_ATTN_HEADS,
                    "DEC_LAYERS": config.DEC_LAYERS,
                    "D_MODEL": config.D_MODEL,
                    "LATENT_LEN": config.LATENT_LEN,
                    "DATA_AUG": config.DATA_AUG})

    elif config.MODEL_TYPE == "FullTransformer":
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.MODEL_TYPE,
            # track hyperparameters and run metadata with wandb.config
            config={"Model": config.MODEL_TYPE,
                    "DATA_PREP_METHOD": config.DATA_PREP_METHOD,
                    "DATA_NORMALIZATION": config.DATA_NORMALIZATION,
                    "LEARNING_RATE": config.LEARNING_RATE,
                    "WITH_WARMUP": config.WITH_WARMUP,
                    "LR_WARMUP": config.LR_WARMUP,
                    "LR_FINAL": config.LR_FINAL,
                    "NUM_EPOCHS": config.NUM_EPOCHS,
                    "BATCH_SIZE": config.BATCH_SIZE,
                    "REG_VALUE": config.REG_VALUE,
                    "DROPOUT_RATE": config.DROPOUT_RATE,
                    "DATA_PREP": config.DATA_PREP,
                    "ENC_DEPTH": config.ENC_DEPTH,
                    "DFF": config.DFF,
                    "NUM_ATTN_HEADS": config.NUM_ATTN_HEADS,
                    "D_MODEL": config.D_MODEL,
                    "LATENT_LEN": config.LATENT_LEN,
                    "DATA_AUG": config.DATA_AUG})

    if config.MODEL_TYPE == "DINO":
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.MODEL_TYPE,
            # track hyperparameters and run metadata with wandb.config
            config={"Model": config.MODEL_TYPE,
                    "DATA_PREP_METHOD": config.DATA_PREP_METHOD,
                    "DATA_NORMALIZATION": config.DATA_NORMALIZATION,
                    "TRAIN_TEST_SPLIT": config.TRAIN_TEST_SPLIT,
                    "LEARNING_RATE": config.LEARNING_RATE,
                    "WITH_WARMUP": config.WITH_WARMUP,
                    "LR_WARMUP": config.LR_WARMUP,
                    "LR_FINAL": config.LR_FINAL,
                    "NUM_EPOCHS": config.NUM_EPOCHS,
                    "BATCH_SIZE": config.BATCH_SIZE,
                    "CENTERING_RATE": config.CENTERING_RATE,
                    "LEARNING_MOMENTUM_RATE": config.LEARNING_MOMENTUM_RATE,
                    "STUDENT_TEMPERATURE": config.STUDENT_TEMPERATURE,
                    "TEACHER_TEMPERATURE": config.TEACHER_TEMPERATURE,
                    "TEACHER_TEMPERATURE_FINAL": config.TEACHER_TEMPERATURE_FINAL,
                    "TEACHER_WARMUP": config.TEACHER_WARMUP,
                    "EMBEDDING_DIM": config.EMBEDDING_DIM,
                    "SEQ_LEN": config.SEQ_LEN,
                    "LATENT_LEN": config.LATENT_LEN,
                    "NUMBER_OF_LAYERS": config.NUMBER_OF_LAYERS,
                    "STATE_INDEX": config.STATE_INDEX,
                    "STATE_CHANNELS": config.STATE_CHANNELS,
                    "DFF": config.DFF,
                    "X_ATTN_HEADS": config.X_ATTN_HEADS,
                    "X_ATTN_DIM": config.X_ATTN_DIM,
                    "DEPTH": config.DEPTH,
                    "NUM_ATTN_HEADS": config.NUM_ATTN_HEADS,
                    "SELF_ATTN_DIM": config.SELF_ATTN_DIM,
                    "DROPOUT_RATE": config.DROPOUT_RATE,
                    "N_CLUSTERS": config.N_CLUSTERS})
