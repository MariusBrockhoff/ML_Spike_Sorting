import wandb


def wandb_initializer(model_config, pretraining_config, finetune_config, method):
    """
    Initialize a Weights & Biases (wandb) run for experiment tracking and logging.

    This function sets up a wandb run based on the provided method (reconstruction, DEC, IDEC)
    and configuration parameters. It logs various hyperparameters and metadata for the model training.

    Arguments:
        model_config: Configuration object containing model specifications.
        pretraining_config: Configuration object containing pretraining specifications.
        finetune_config: Configuration object containing finetuning specifications.
        method: String indicating the training method ('reconstruction', 'DEC', or 'IDEC').

    The function initializes a wandb run with a project name and configuration parameters specific to the chosen method.
    The project name and parameters are set based on the model type, method, and configurations provided.

    For 'reconstruction' method:
        - The project is named "Pretraining_{method}_{MODEL_TYPE}".
        - Configurations related to pretraining and model settings are logged.

    For 'NNCLR' method:
        - The project is named "Pretraining_{method}_{MODEL_TYPE}".
        - Configurations related to pretraining and model settings are logged.

    For 'PseudoLabel' method:
        - The project is named "Finetuning_PseudoLabel_{MODEL_TYPE}".
        - Configurations related to finetuning and model settings are logged.
    """

    if method == "reconstruction":
        if model_config.MODEL_TYPE == "DenseAutoencoder":
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

    elif method == "NNCLR":
        if model_config.MODEL_TYPE == "DenseAutoencoder":
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

    elif method == "PseudoLabel":
        wandb.init(
            # set the wandb project where this run will be logged
            project="Finetuning_PseudoLabel_" + model_config.MODEL_TYPE,
            # track hyperparameters and run metadata with wandb.model_config
            config={"Model": model_config.MODEL_TYPE,
                    "DATA": finetune_config.FILE_NAME,
                    "PSEUDO_N_CLUSTERS": finetune_config.PSEUDO_N_CLUSTERS,
                    "PSEUDO_LABEL_RATIO": finetune_config.PSEUDO_LABEL_RATIO,
                    "ITERATIVE_RATIOS": finetune_config.ITERATIVE_RATIOS,
                    "SAMPLING_METHOD": finetune_config.SAMPLING_METHOD,
                    "DENSITY_FUNCTION": finetune_config.DENSITY_FUNCTION,
                    "K_NEAREST_NEIGHBOURS": finetune_config.K_NEAREST_NEIGHBOURS,
                    "PSEUDO_EPOCHS": finetune_config.PSEUDO_EPOCHS,
                    "PSEUDO_BATCH_SIZE": finetune_config.PSEUDO_BATCH_SIZE})
