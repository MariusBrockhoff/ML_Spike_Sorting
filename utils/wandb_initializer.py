# -*- coding: utf-8 -*-
"""
Load Spike data and prepare for training of model
"""
import wandb

def wandb_initializer(config, method, fine_tune_config):
    if method == "reconstruction":
            if config.MODEL_TYPE == "PerceiverIO":
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Pretraining_Reconstruction_" + config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.config
                    config={"Model": config.MODEL_TYPE,
                            "DATA": config.FILE_NAME,
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
                        project="Pretraining_Reconstruction_" + config.MODEL_TYPE,
                        # track hyperparameters and run metadata with wandb.config
                        config={"Model": config.MODEL_TYPE,
                                "DATA": config.FILE_NAME,
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


            elif config.MODEL_TYPE == "AttnE_DenseD":
                    wandb.init(
                            # set the wandb project where this run will be logged
                            project="Pretraining_Reconstruction_" + config.MODEL_TYPE,
                            # track hyperparameters and run metadata with wandb.config
                            config={"Model": config.MODEL_TYPE,
                                    "DATA": config.FILE_NAME,
                                    "DATA_PREP_METHOD": config.DATA_PREP_METHOD,
                                    "DATA_NORMALIZATION": config.DATA_NORMALIZATION,
                                    "LEARNING_RATE": config.LEARNING_RATE,
                                    "WITH_WARMUP": config.WITH_WARMUP,
                                    "LR_WARMUP": config.LR_WARMUP,
                                    "LR_FINAL": config.LR_FINAL,
                                    "NUM_EPOCHS": config.NUM_EPOCHS,
                                    "BATCH_SIZE": config.BATCH_SIZE,
                                    "LATENT_LEN": config.LATENT_LEN,
                                    "Embedding_DIM": config.Embedding_dim,
                                    "DFF": config.DFF,
                                    "ENC_DEPTH": config.ENC_DEPTH,
                                    "ENC_NUM_ATTN_HEADS": config.ENC_NUM_ATTN_HEADS,
                                    "ENC_DROPOUT_RATE": config.ENC_DROPOUT_RATE,
                                    "DEC_LAYERS": config.DEC_LAYERS})

            elif config.MODEL_TYPE == "FullTransformer":
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Pretraining_Reconstruction_" + config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.config
                    config={"Model": config.MODEL_TYPE,
                            "DATA": config.FILE_NAME,
                            "DATA_PREP_METHOD": config.DATA_PREP_METHOD,
                            "DATA_NORMALIZATION": config.DATA_NORMALIZATION,
                            "LEARNING_RATE": config.LEARNING_RATE,
                            "WITH_WARMUP": config.WITH_WARMUP,
                            "LR_WARMUP": config.LR_WARMUP,
                            "LR_FINAL": config.LR_FINAL,
                            "NUM_EPOCHS": config.NUM_EPOCHS,
                            "BATCH_SIZE": config.BATCH_SIZE,
                            "LATENT_LEN": config.LATENT_LEN,
                            "Embedding_DIM": config.Embedding_dim,
                            "DFF": config.DFF,
                            "ENC_DEPTH": config.ENC_DEPTH,
                            "ENC_NUM_ATTN_HEADS": config.ENC_NUM_ATTN_HEADS,
                            "ENC_DROPOUT_RATE": config.ENC_DROPOUT_RATE,
                            "DEC_DEPTH": config.DEC_DEPTH,
                            "DEC_NUM_ATTN_HEADS": config.DEC_NUM_ATTN_HEADS,
                            "DEC_DROPOUT_RATE": config.DEC_DROPOUT_RATE,
                            "DATA_AUG": config.DATA_AUG})

    elif method == "NNCLR":
        #TODO: implement NNCLR pretraining
        print("NNCLR still to be implemented")

    elif method == "DINO":
            if config.MODEL_TYPE == "DINO":
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Pretraining_DINO_" + config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.config
                    config={"Model": config.MODEL_TYPE,
                            "DATA": config.FILE_NAME,
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

    elif method == "DEC":
            wandb.init(
                # set the wandb project where this run will be logged
                project="Finetuning_DEC_" + config.MODEL_TYPE,
                # track hyperparameters and run metadata with wandb.config
                config={"Model": config.MODEL_TYPE,
                        "DATA": config.FILE_NAME,
                        "DEC_N_CLUSTERS": fine_tune_config.DEC_N_CLUSTERS,
                        "DEC_BATCH_SIZE": fine_tune_config.DEC_BATCH_SIZE,
                        "DEC_LEARNING_RATE": fine_tune_config.DEC_LEARNING_RATE,
                        "DEC_MOMENTUM": fine_tune_config.DEC_MOMENTUM,
                        "DEC_TOL": fine_tune_config.DEC_TOL,
                        "DEC_MAXITER": fine_tune_config.DEC_MAXITER,
                        "DEC_UPDATE_INTERVAL": fine_tune_config.DEC_UPDATE_INTERVAL,
                        "DEC_SAVE_DIR": fine_tune_config.DEC_SAVE_DIR})

    elif method == "IDEC":
            wandb.init(
                # set the wandb project where this run will be logged
                project="Finetuning_IDEC_" + config.MODEL_TYPE,
                # track hyperparameters and run metadata with wandb.config
                config={"Model": config.MODEL_TYPE,
                        "DATA": config.FILE_NAME,
                        "IDEC_N_CLUSTERS": fine_tune_config.IDEC_N_CLUSTERS,
                        "IDEC_BATCH_SIZE": fine_tune_config.IDEC_BATCH_SIZE,
                        "IDEC_LEARNING_RATE": fine_tune_config.IDEC_LEARNING_RATE,
                        "IDEC_MOMENTUM": fine_tune_config.IDEC_MOMENTUM,
                        "IDEC_GAMMA": fine_tune_config.IDEC_GAMMA,
                        "IDEC_TOL": fine_tune_config.IDEC_TOL,
                        "IDEC_MAXITER": fine_tune_config.IDEC_MAXITER,
                        "IDEC_UPDATE_INTERVAL": fine_tune_config.IDEC_UPDATE_INTERVAL,
                        "IDEC_SAVE_DIR": fine_tune_config.IDEC_SAVE_DIR})

    elif method == "PseudoLabel":
            wandb.init(
                    # set the wandb project where this run will be logged
                    project="Finetuning_PseudoLabel_" + config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.config
                    config={"Model": config.MODEL_TYPE,
                            "DATA": config.FILE_NAME,
                            "PSEUDO_N_CLUSTERS": fine_tune_config.PSEUDO_N_CLUSTERS,
                            "PSEUDO_EPOCHS": fine_tune_config.PSEUDO_EPOCHS,
                            "PSEUDO_BATCH_SIZE": fine_tune_config.PSEUDO_BATCH_SIZE,
                            "PSEUDO_RATIO": fine_tune_config.PSEUDO_RATIO})
