# -*- coding: utf-8 -*-
"""
Load Spike data and prepare for training of model
"""
import wandb

def wandb_initializer(model_config, pretraining_config, fintune_config, method):
    if method == "reconstruction":
            if model_config.MODEL_TYPE == "PerceiverIO":
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.model_config
                    config={"Model": model_config.MODEL_TYPE,
                            "DATA": pretraining_config.FILE_NAME,
                            "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                            "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                            "TRAIN_TEST_SPLIT": pretraining_config.TRAIN_TEST_SPLIT,
                            "LEARNING_RATE": pretraining_config.LEARNING_RATE,
                            "WITH_WARMUP": pretraining_config.WITH_WARMUP,
                            "LR_WARMUP": pretraining_config.LR_WARMUP,
                            "LR_FINAL": pretraining_config.LR_FINAL,
                            "NUM_EPOCHS": pretraining_config.NUM_EPOCHS,
                            "BATCH_SIZE": pretraining_config.BATCH_SIZE,
                            "EMBEDDING_DIM": model_config.EMBEDDING_DIM,
                            "SEQ_LEN": model_config.SEQ_LEN,
                            "LATENT_LEN": model_config.LATENT_LEN,
                            "ENC_NUMBER_OF_LAYERS": model_config.ENC_NUMBER_OF_LAYERS,
                            "ENC_STATE_INDEX": model_config.ENC_STATE_INDEX,
                            "ENC_STATE_CHANNELS": model_config.ENC_STATE_CHANNELS,
                            "ENC_DFF": model_config.ENC_DFF,
                            "ENC_X_ATTN_HEADS": model_config.ENC_X_ATTN_HEADS,
                            "ENC_X_ATTN_DIM": model_config.ENC_X_ATTN_DIM,
                            "ENC_DEPTH": model_config.ENC_DEPTH,
                            "ENC_NUM_ATTN_HEADS": model_config.ENC_NUM_ATTN_HEADS,
                            "ENC_SELF_ATTN_DIM": model_config.ENC_SELF_ATTN_DIM,
                            "ENC_DROPOUT_RATE": model_config.ENC_DROPOUT_RATE,
                            "DEC_NUMBER_OF_LAYERS": model_config.DEC_NUMBER_OF_LAYERS,
                            "DEC_STATE_INDEX": model_config.DEC_STATE_INDEX,
                            "DEC_STATE_CHANNELS": model_config.DEC_STATE_CHANNELS,
                            "DEC_DFF": model_config.DEC_DFF,
                            "DEC_X_ATTN_HEADS": model_config.DEC_X_ATTN_HEADS,
                            "DEC_X_ATTN_DIM": model_config.DEC_X_ATTN_DIM,
                            "DEC_DEPTH": model_config.DEC_DEPTH,
                            "DEC_NUM_ATTN_HEADS": model_config.DEC_NUM_ATTN_HEADS,
                            "DEC_SELF_ATTN_DIM": model_config.DEC_SELF_ATTN_DIM,
                            "DEC_DROPOUT_RATE": model_config.DEC_DROPOUT_RATE,
                            "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                            "N_CLUSTERS": pretraining_config.N_CLUSTERS})

            elif model_config.MODEL_TYPE == "DenseAutoencoder":
                wandb.init(
                        # set the wandb project where this run will be logged
                        project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                        # track hyperparameters and run metadata with wandb.model_config
                        config={"Model": model_config.MODEL_TYPE,
                                "DATA": pretraining_config.FILE_NAME,
                                "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                                "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                                "LEARNING_RATE": pretraining_config.LEARNING_RATE,
                                "WITH_WARMUP": pretraining_config.WITH_WARMUP,
                                "LR_WARMUP": pretraining_config.LR_WARMUP,
                                "LR_FINAL": pretraining_config.LR_FINAL,
                                "NUM_EPOCHS": pretraining_config.NUM_EPOCHS,
                                "BATCH_SIZE": pretraining_config.BATCH_SIZE,
                                "DIMS": model_config.DIMS,
                                "ACT": model_config.ACT,
                                "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                                "N_CLUSTERS": pretraining_config.N_CLUSTERS})


            elif model_config.MODEL_TYPE == "AttnE_DenseD":
                    wandb.init(
                            # set the wandb project where this run will be logged
                            project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                            # track hyperparameters and run metadata with wandb.model_config
                            config={"Model": model_config.MODEL_TYPE,
                                    "DATA": pretraining_config.FILE_NAME,
                                    "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                                    "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                                    "LEARNING_RATE": pretraining_config.LEARNING_RATE,
                                    "WITH_WARMUP": pretraining_config.WITH_WARMUP,
                                    "LR_WARMUP": pretraining_config.LR_WARMUP,
                                    "LR_FINAL": pretraining_config.LR_FINAL,
                                    "NUM_EPOCHS": pretraining_config.NUM_EPOCHS,
                                    "BATCH_SIZE": pretraining_config.BATCH_SIZE,
                                    "LATENT_LEN": model_config.LATENT_LEN,
                                    "Embedding_DIM": model_config.EMBEDDING_DIM,
                                    "DFF": model_config.DFF,
                                    "ENC_DEPTH": model_config.ENC_DEPTH,
                                    "ENC_NUM_ATTN_HEADS": model_config.ENC_NUM_ATTN_HEADS,
                                    "ENC_DROPOUT_RATE": model_config.ENC_DROPOUT_RATE,
                                    "DEC_LAYERS": model_config.DEC_LAYERS,
                                    "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                                    "N_CLUSTERS": pretraining_config.N_CLUSTERS})

            elif model_config.MODEL_TYPE == "FullTransformer":
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.model_config
                    config={"Model": model_config.MODEL_TYPE,
                            "DATA": pretraining_config.FILE_NAME,
                            "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                            "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                            "LEARNING_RATE": pretraining_config.LEARNING_RATE,
                            "WITH_WARMUP": pretraining_config.WITH_WARMUP,
                            "LR_WARMUP": pretraining_config.LR_WARMUP,
                            "LR_FINAL": pretraining_config.LR_FINAL,
                            "NUM_EPOCHS": pretraining_config.NUM_EPOCHS,
                            "BATCH_SIZE": pretraining_config.BATCH_SIZE,
                            "LATENT_LEN": model_config.LATENT_LEN,
                            "Embedding_DIM": model_config.EMBEDDING_DIM,
                            "DFF": model_config.DFF,
                            "ENC_DEPTH": model_config.ENC_DEPTH,
                            "ENC_NUM_ATTN_HEADS": model_config.ENC_NUM_ATTN_HEADS,
                            "ENC_DROPOUT_RATE": model_config.ENC_DROPOUT_RATE,
                            "DEC_DEPTH": model_config.DEC_DEPTH,
                            "DEC_NUM_ATTN_HEADS": model_config.DEC_NUM_ATTN_HEADS,
                            "DEC_DROPOUT_RATE": model_config.DEC_DROPOUT_RATE,
                            "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                            "N_CLUSTERS": pretraining_config.N_CLUSTERS})

    elif method == "NNCLR":
            if model_config.MODEL_TYPE == "PerceiverIO":
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.model_config
                    config={"Model": model_config.MODEL_TYPE,
                            "DATA": pretraining_config.FILE_NAME,
                            "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                            "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                            "TRAIN_TEST_SPLIT": pretraining_config.TRAIN_TEST_SPLIT,
                            "LEARNING_RATE": pretraining_config.LEARNING_RATE_NNCLR,
                            "WITH_WARMUP": pretraining_config.WITH_WARMUP_NNCLR,
                            "LR_WARMUP": pretraining_config.LR_WARMUP_NNCLR,
                            "LR_FINAL": pretraining_config.LR_FINAL_NNCLR,
                            "NUM_EPOCHS": pretraining_config.NUM_EPOCHS_NNCLR,
                            "BATCH_SIZE": pretraining_config.BATCH_SIZE_NNCLR,
                            "TEMPERATURE": pretraining_config.TEMPERATURE,
                            "QUEUE_SIZE": pretraining_config.QUEUE_SIZE,
                            "PROJECTION_WIDTH": pretraining_config.PROJECTION_WIDTH,
                            "EMBEDDING_DIM": model_config.EMBEDDING_DIM,
                            "SEQ_LEN": model_config.SEQ_LEN,
                            "LATENT_LEN": model_config.LATENT_LEN,
                            "ENC_NUMBER_OF_LAYERS": model_config.ENC_NUMBER_OF_LAYERS,
                            "ENC_STATE_INDEX": model_config.ENC_STATE_INDEX,
                            "ENC_STATE_CHANNELS": model_config.ENC_STATE_CHANNELS,
                            "ENC_DFF": model_config.ENC_DFF,
                            "ENC_X_ATTN_HEADS": model_config.ENC_X_ATTN_HEADS,
                            "ENC_X_ATTN_DIM": model_config.ENC_X_ATTN_DIM,
                            "ENC_DEPTH": model_config.ENC_DEPTH,
                            "ENC_NUM_ATTN_HEADS": model_config.ENC_NUM_ATTN_HEADS,
                            "ENC_SELF_ATTN_DIM": model_config.ENC_SELF_ATTN_DIM,
                            "ENC_DROPOUT_RATE": model_config.ENC_DROPOUT_RATE,
                            "DEC_NUMBER_OF_LAYERS": model_config.DEC_NUMBER_OF_LAYERS,
                            "DEC_STATE_INDEX": model_config.DEC_STATE_INDEX,
                            "DEC_STATE_CHANNELS": model_config.DEC_STATE_CHANNELS,
                            "DEC_DFF": model_config.DEC_DFF,
                            "DEC_X_ATTN_HEADS": model_config.DEC_X_ATTN_HEADS,
                            "DEC_X_ATTN_DIM": model_config.DEC_X_ATTN_DIM,
                            "DEC_DEPTH": model_config.DEC_DEPTH,
                            "DEC_NUM_ATTN_HEADS": model_config.DEC_NUM_ATTN_HEADS,
                            "DEC_SELF_ATTN_DIM": model_config.DEC_SELF_ATTN_DIM,
                            "DEC_DROPOUT_RATE": model_config.DEC_DROPOUT_RATE,
                            "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                            "N_CLUSTERS": pretraining_config.N_CLUSTERS})

            elif model_config.MODEL_TYPE == "DenseAutoencoder":
                wandb.init(
                        # set the wandb project where this run will be logged
                        project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                        # track hyperparameters and run metadata with wandb.model_config
                        config={"Model": model_config.MODEL_TYPE,
                                "DATA": pretraining_config.FILE_NAME,
                                "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                                "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                                "LEARNING_RATE": pretraining_config.LEARNING_RATE_NNCLR,
                                "WITH_WARMUP": pretraining_config.WITH_WARMUP_NNCLR,
                                "LR_WARMUP": pretraining_config.LR_WARMUP_NNCLR,
                                "LR_FINAL": pretraining_config.LR_FINAL_NNCLR,
                                "NUM_EPOCHS": pretraining_config.NUM_EPOCHS_NNCLR,
                                "BATCH_SIZE": pretraining_config.BATCH_SIZE_NNCLR,
                                "TEMPERATURE": pretraining_config.TEMPERATURE,
                                "QUEUE_SIZE": pretraining_config.QUEUE_SIZE,
                                "PROJECTION_WIDTH": pretraining_config.PROJECTION_WIDTH,
                                "DIMS": model_config.DIMS,
                                "ACT": model_config.ACT,
                                "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                                "N_CLUSTERS": pretraining_config.N_CLUSTERS})


            elif model_config.MODEL_TYPE == "AttnE_DenseD":
                    wandb.init(
                            # set the wandb project where this run will be logged
                            project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                            # track hyperparameters and run metadata with wandb.model_config
                            config={"Model": model_config.MODEL_TYPE,
                                    "DATA": pretraining_config.FILE_NAME,
                                    "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                                    "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                                    "LEARNING_RATE": pretraining_config.LEARNING_RATE_NNCLR,
                                    "WITH_WARMUP": pretraining_config.WITH_WARMUP_NNCLR,
                                    "LR_WARMUP": pretraining_config.LR_WARMUP_NNCLR,
                                    "LR_FINAL": pretraining_config.LR_FINAL_NNCLR,
                                    "NUM_EPOCHS": pretraining_config.NUM_EPOCHS_NNCLR,
                                    "BATCH_SIZE": pretraining_config.BATCH_SIZE_NNCLR,
                                    "TEMPERATURE": pretraining_config.TEMPERATURE,
                                    "QUEUE_SIZE": pretraining_config.QUEUE_SIZE,
                                    "PROJECTION_WIDTH": pretraining_config.PROJECTION_WIDTH,
                                    "LATENT_LEN": model_config.LATENT_LEN,
                                    "Embedding_DIM": model_config.EMBEDDING_DIM,
                                    "DFF": model_config.DFF,
                                    "ENC_DEPTH": model_config.ENC_DEPTH,
                                    "ENC_NUM_ATTN_HEADS": model_config.ENC_NUM_ATTN_HEADS,
                                    "ENC_DROPOUT_RATE": model_config.ENC_DROPOUT_RATE,
                                    "DEC_LAYERS": model_config.DEC_LAYERS,
                                    "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                                    "N_CLUSTERS": pretraining_config.N_CLUSTERS})

            elif model_config.MODEL_TYPE == "FullTransformer":
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Pretraining_" + method + "_" + model_config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.model_config
                    config={"Model": model_config.MODEL_TYPE,
                            "DATA": pretraining_config.FILE_NAME,
                            "DATA_PREP_METHOD": pretraining_config.DATA_PREP_METHOD,
                            "DATA_NORMALIZATION": pretraining_config.DATA_NORMALIZATION,
                            "LEARNING_RATE": pretraining_config.LEARNING_RATE_NNCLR,
                            "WITH_WARMUP": pretraining_config.WITH_WARMUP_NNCLR,
                            "LR_WARMUP": pretraining_config.LR_WARMUP_NNCLR,
                            "LR_FINAL": pretraining_config.LR_FINAL_NNCLR,
                            "NUM_EPOCHS": pretraining_config.NUM_EPOCHS_NNCLR,
                            "BATCH_SIZE": pretraining_config.BATCH_SIZE_NNCLR,
                            "TEMPERATURE": pretraining_config.TEMPERATURE,
                            "QUEUE_SIZE": pretraining_config.QUEUE_SIZE,
                            "PROJECTION_WIDTH": pretraining_config.PROJECTION_WIDTH,
                            "LATENT_LEN": model_config.LATENT_LEN,
                            "Embedding_DIM": model_config.EMBEDDING_DIM,
                            "DFF": model_config.DFF,
                            "ENC_DEPTH": model_config.ENC_DEPTH,
                            "ENC_NUM_ATTN_HEADS": model_config.ENC_NUM_ATTN_HEADS,
                            "ENC_DROPOUT_RATE": model_config.ENC_DROPOUT_RATE,
                            "DEC_DEPTH": model_config.DEC_DEPTH,
                            "DEC_NUM_ATTN_HEADS": model_config.DEC_NUM_ATTN_HEADS,
                            "DEC_DROPOUT_RATE": model_config.DEC_DROPOUT_RATE,
                            "CLUSTERING_METHOD": pretraining_config.CLUSTERING_METHOD,
                            "N_CLUSTERS": pretraining_config.N_CLUSTERS})


    elif method == "DEC":
            wandb.init(
                # set the wandb project where this run will be logged
                project="Finetuning_DEC_" + model_config.MODEL_TYPE,
                # track hyperparameters and run metadata with wandb.model_config
                config={"Model": model_config.MODEL_TYPE,
                        "DATA": fintune_config.FILE_NAME,
                        "DEC_N_CLUSTERS": fintune_config.DEC_N_CLUSTERS,
                        "DEC_BATCH_SIZE": fintune_config.DEC_BATCH_SIZE,
                        "DEC_LEARNING_RATE": fintune_config.DEC_LEARNING_RATE,
                        "DEC_MOMENTUM": fintune_config.DEC_MOMENTUM,
                        "DEC_TOL": fintune_config.DEC_TOL,
                        "DEC_MAXITER": fintune_config.DEC_MAXITER,
                        "DEC_UPDATE_INTERVAL": fintune_config.DEC_UPDATE_INTERVAL,
                        "DEC_SAVE_DIR": fintune_config.DEC_SAVE_DIR})

    elif method == "IDEC":
            wandb.init(
                # set the wandb project where this run will be logged
                project="Finetuning_IDEC_" + model_config.MODEL_TYPE,
                # track hyperparameters and run metadata with wandb.model_config
                config={"Model": model_config.MODEL_TYPE,
                        "DATA": fintune_config.FILE_NAME,
                        "IDEC_N_CLUSTERS": fintune_config.IDEC_N_CLUSTERS,
                        "IDEC_BATCH_SIZE": fintune_config.IDEC_BATCH_SIZE,
                        "IDEC_LEARNING_RATE": fintune_config.IDEC_LEARNING_RATE,
                        "IDEC_MOMENTUM": fintune_config.IDEC_MOMENTUM,
                        "IDEC_GAMMA": fintune_config.IDEC_GAMMA,
                        "IDEC_TOL": fintune_config.IDEC_TOL,
                        "IDEC_MAXITER": fintune_config.IDEC_MAXITER,
                        "IDEC_UPDATE_INTERVAL": fintune_config.IDEC_UPDATE_INTERVAL,
                        "IDEC_SAVE_DIR": fintune_config.IDEC_SAVE_DIR})

    elif method == "PseudoLabel":
            wandb.init(
                    # set the wandb project where this run will be logged
                    project="Finetuning_PseudoLabel_" + model_config.MODEL_TYPE,
                    # track hyperparameters and run metadata with wandb.model_config
                    config={"Model": model_config.MODEL_TYPE,
                            "DATA": fintune_config.FILE_NAME,
                            "PSEUDO_N_CLUSTERS": fintune_config.PSEUDO_N_CLUSTERS,
                            "PSEUDO_LABEL_RATIO": fintune_config.PSEUDO_LABEL_RATIO,
                            "ITERATIVE_RATIOS": fintune_config.ITERATIVE_RATIOS,
                            "SAMPLING_METHOD": fintune_config.SAMPLING_METHOD,
                            "DENSITY_FUNCTION": fintune_config.DENSITY_FUNCTION,
                            "K_NEAREST_NEIGHBOURS": fintune_config.K_NEAREST_NEIGHBOURS,
                            "PSEUDO_EPOCHS": fintune_config.PSEUDO_EPOCHS,
                            "PSEUDO_BATCH_SIZE": fintune_config.PSEUDO_BATCH_SIZE})
