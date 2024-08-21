import tensorflow as tf
import tensorflow_addons as tfa

from os import path
from utils.evaluation import *


def check_filepath_naming(filepath):
    """
    Check and modify the file path to avoid overwriting existing files.

    If the specified file path exists, a number is appended to the file name to
    create a unique file name.

    Arguments:
        filepath: The file path to check.

    Returns:
        A modified file path if the original exists, otherwise the original file path.
    """
    if path.exists(filepath):
        numb = 1
        while True:
            new_path = "{0}_{2}{1}".format(*path.splitext(filepath) + (numb,))
            if path.exists(new_path):
                numb += 1
            else:
                return new_path
    return filepath


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    """
    Generate a cosine learning rate schedule with optional warmup.

    Arguments:
        base_value: The initial learning rate or weight decay value.
        final_value: The final learning rate or weight decay value.
        epochs: Total number of epochs.
        warmup_epochs: Number of warmup epochs. Default is 0.
        start_warmup_value: Starting value for warmup. Default is 0.

    Returns:
        A numpy array containing the learning rate or weight decay value for each epoch.
    """
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs)

    epoch_counter = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(epoch_counter * np.pi / epochs))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule


class EarlyStopper:
    """
    Early stopping utility to stop training when a certain criteria is met.

    Arguments:
        patience: Number of epochs to wait for improvement before stopping. Default is 5.
        min_delta: Minimum change to quantify as an improvement. Default is 0.
        baseline: Baseline value for comparison. Default is 0.

    Methods:
        early_stop(validation_loss): Determine if training should be stopped based on the validation loss.
    """
    def __init__(self, patience=5, min_delta=0, baseline=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.baseline = baseline
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if (validation_loss + self.min_delta) < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        if validation_loss < self.baseline:
            return True
        return False


class RandomNoise(tf.keras.layers.Layer):
    """
    A custom Keras layer that adds random noise to its inputs.

    This layer is designed to augment data by adding random Gaussian noise. The level of noise can be adjusted,
    and the application of noise can be toggled on or off. This type of layer is often used in training deep learning
    models to improve robustness and prevent overfitting, as it introduces variability in the training data.

    Attributes:
    apply_noise (bool): If True, noise is applied to the inputs; if False, the layer acts as a no-op.
    max_noise_lvl (float): The maximum standard deviation of the Gaussian noise to be added.
    seed (int, optional): Seed for random number generation to ensure reproducibility.

    Methods:
    build(input_shape): Builds the layer (no trainable weights).
    call(inputs): Logic for adding noise to the inputs.

    The layer, when called on input data, generates noise with a random level (up to `max_noise_lvl`) and adds it to the
    input.
    If `apply_noise` is set to False, the layer simply passes the input data through without modification.
    """

    def __init__(self, apply_noise=True, max_noise_lvl=1.0, seed=None):
        """
        Initializes the RandomNoise layer.

        Parameters:
        apply_noise (bool): Whether to apply noise or not. Defaults to True.
        max_noise_lvl (float): Maximum level of noise to apply. Defaults to 1.0.
        seed (int, optional): Seed for random number generation. Defaults to None.
        """
        super().__init__()
        self.apply_noise = apply_noise
        self.max_noise_lvl = max_noise_lvl
        self.seed = seed

    def build(self, input_shape):
        """
        Build method for the layer.

        This layer does not have any trainable weights, so the method simply calls the parent's build method.

        Parameters:
        input_shape (tuple of int): The shape of the input to the layer.
        """
        super().build(input_shape)

    def call(self, inputs):
        """
        The logic for adding random noise to the inputs.

        If `apply_noise` is True, it generates and adds Gaussian noise to the input. The standard deviation of the noise
        is randomly chosen between 0 and `max_noise_lvl` for each forward pass through the layer. If `apply_noise` is
        False, inputs are passed through unchanged.

        Parameters:
        inputs (tf.Tensor): The input tensor to the layer.

        Returns:
        tf.Tensor: The input tensor, with random noise added if `apply_noise` is True.
        """
        if self.apply_noise:
            noise_lvl = tf.random.uniform(
                shape=(),
                minval=0,
                maxval=self.max_noise_lvl,
                dtype=tf.float32,
                seed=self.seed
            )

            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=noise_lvl,
                dtype=tf.float32,
                seed=self.seed
            )

            inputs += noise  # Adding generated noise to the inputs

        return inputs


def spikeaugmentation(apply_noise, max_noise_lvl, scale, name):
    """
    Creates a sequential model for data augmentation specifically designed for spike data.

    This function constructs a Keras Sequential model consisting of layers that apply various augmentations to the input
    data. Currently, it includes the RandomNoise layer for adding Gaussian noise. This augmentation technique is useful
    in training neural network models, especially when dealing with spike data, as it helps in enhancing the model's
    robustness and generalization by introducing variability in the training data.

    Parameters:
    apply_noise (bool): Whether to apply random noise to the data.
    max_noise_lvl (float): The maximum level of noise to be added.
    scale (tuple): The scaling factor to be used in the RandomResizedCrop1D layer (currently not included in the
                   Sequential model).
    name (str): The name of the Sequential model.

    Returns:
    tf.keras.Sequential: A Keras Sequential model with specified augmentation layers.

    The function constructs a Sequential model comprising a RandomNoise layer. The RandomNoise layer's behavior is
    controlled by the 'apply_noise' and 'max_noise_lvl' parameters. The 'scale' parameter is reserved for future use
    with a potential RandomResizedCrop1D layer. The 'name' parameter assigns a name to the Sequential model for
    identification.
    """
    return tf.keras.Sequential(
        [
            # tf.keras.Input(shape=(63,)),
            RandomNoise(apply_noise=apply_noise, max_noise_lvl=max_noise_lvl),
            # RandomResizedCrop1D(scale=scale, seq_len=input_shape[0]),
        ],
        name=name,
    )


class NNCLR(tf.keras.Model):
    """
    NNCLR (Neural Network Contrastive Learning Representation) is a custom Keras Model for semi-supervised learning.

    This class implements a NNCLR model that uses contrastive learning for feature extraction and a linear probe
    for classification. It is designed for scenarios where labeled data is scarce, and the model needs to learn
    useful representations from unlabeled data. The model consists of an encoder, a projection head for contrastive
    learning, and a linear probe for classification.

    Attributes:
    model_config: Configuration object containing model-specific parameters.
    pretraining_config: Configuration object containing pretraining-specific parameters.
    encoder: The neural network used to encode the input data.
    contrastive_augmenter: A function or layer that applies augmentation for contrastive learning.
    classification_augmenter: A function or layer that applies augmentation for classification.
    probe_accuracy, correlation_accuracy, contrastive_accuracy: Keras metrics for tracking training performance.
    probe_loss: Loss function for the probe classifier.
    projection_head: A Keras Sequential model used in the projection of encoded features.
    linear_probe: A Keras Sequential model for classification.
    temperature: The temperature parameter used in contrastive loss calculation.
    feature_queue: A queue of features used in contrastive learning for negative sampling.

    Methods:
    compile(contrastive_optimizer, probe_optimizer, **kwargs): Compiles the NNCLR model with given optimizers.
    nearest_neighbour(projections): Computes nearest neighbours in the feature space for contrastive learning.
    update_contrastive_accuracy(features_1, features_2): Updates the contrastive accuracy metric.
    update_correlation_accuracy(features_1, features_2): Updates the correlation accuracy metric.
    contrastive_loss(projections_1, projections_2): Computes the contrastive loss between two sets of projections.
    train_step(data): Defines a single step of training.
    test_step(data): Defines a single step of testing.

    The NNCLR model uses contrastive learning to learn representations by maximizing agreement between differently
    augmented views of the same data instance and uses a linear classifier (probe) to perform classification based on
    these representations.
    """
    def __init__(self, model_config, pretraining_config, encoder, contrastive_augmenter, classification_augmenter):
        super().__init__()
        self.probe_optimizer = None
        self.contrastive_optimizer = None
        self.model_config = model_config
        self.pretraining_config = pretraining_config
        self.probe_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.probe_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.projection_width = self.pretraining_config.PROJECTION_WIDTH
        self.contrastive_augmenter = spikeaugmentation(**contrastive_augmenter)
        self.classification_augmenter = spikeaugmentation(**classification_augmenter)
        self.encoder = encoder
        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.model_config.LATENT_LEN,)),
                tf.keras.layers.Dense(self.projection_width, activation="relu"),
                tf.keras.layers.Dense(self.projection_width),
            ],
            name="projection_head",
        )
        self.linear_probe = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(self.projection_width,)),
             tf.keras.layers.Dense(pretraining_config.N_CLUSTERS)], name="linear_probe"
        )
        self.temperature = self.pretraining_config.TEMPERATURE

        feature_dimensions = self.model_config.LATENT_LEN
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(self.pretraining_config.QUEUE_SIZE, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

    def nearest_neighbour(self, projections):
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.shape(features_1, out_type=tf.int32)[0]
        batch_size = tf.cast(batch_size, dtype=tf.float32, name=None)
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            tf.concat(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0,
            ),
            from_logits=True,
        )

        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)
        preprocessed_images = self.classification_augmenter(labeled_images)

        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labels, class_logits)

        res = {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
        }

        return res

    def test_step(self, data):
        labeled_images, labels = data

        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        self.probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()}


def pretrain_model(model, model_config, pretraining_config, pretrain_method, dataset, dataset_test, save_weights,
                   save_dir):
    """
    Pretrains a model using a specified method and configuration on given datasets.

    This function supports different pretraining methods, particularly 'reconstruction' and 'NNCLR' (Neural Network
    Contrastive Learning Representation). It handles the entire pretraining process, including setting up the optimizer,
    loss functions, and learning rate schedules. It also provides support for early stopping and logs training metrics
    using wandb.

    Parameters:
    model (tf.keras.Model): The neural network model to be pretrained.
    model_config: Configuration object containing model-specific parameters.
    pretraining_config: Configuration object containing pretraining-specific parameters.
    pretrain_method (str): The method used for pretraining ('reconstruction', 'NNCLR', etc.).
    dataset (tf.data.Dataset): The training dataset.
    dataset_test (tf.data.Dataset): The test dataset.
    save_weights (bool): Whether to save the pretrained weights.
    save_dir (str): Directory where the pretrained weights should be saved.

    The function first checks the pretraining method. For 'reconstruction', it uses a Mean Squared Error loss to train
    the model to reconstruct its inputs. For 'NNCLR', it sets up a NNCLR model and trains it using contrastive learning.
    The function supports early stopping based on validation loss and learning rate scheduling. After training, it saves
    the model weights if 'save_weights' is True.

    Returns:
    str: The path where the pretrained model's weights are saved.
    """

    if pretrain_method == "reconstruction":
        if pretraining_config.EARLY_STOPPING:
            early_stopper = EarlyStopper(patience=pretraining_config.PATIENCE, min_delta=pretraining_config.MIN_DELTA,
                                         baseline=pretraining_config.BASELINE)

        if pretraining_config.WITH_WARMUP:
            lr_schedule = cosine_scheduler(pretraining_config.LEARNING_RATE,
                                           pretraining_config.LR_FINAL,
                                           pretraining_config.NUM_EPOCHS,
                                           warmup_epochs=pretraining_config.LR_WARMUP,
                                           start_warmup_value=0)
        else:
            lr_schedule = cosine_scheduler(pretraining_config.LEARNING_RATE,
                                           pretraining_config.LR_FINAL,
                                           pretraining_config.NUM_EPOCHS,
                                           warmup_epochs=0,
                                           start_warmup_value=0)
        wd_schedule = cosine_scheduler(pretraining_config.WEIGHT_DECAY,
                                       pretraining_config.WD_FINAL,
                                       pretraining_config.NUM_EPOCHS,
                                       warmup_epochs=0,
                                       start_warmup_value=0)
        if pretraining_config.WITH_WD:
            optimizer = tfa.optimizers.AdamW(weight_decay=wd_schedule[0], learning_rate=lr_schedule[0])
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule[0])

        loss_lst = []
        test_loss_lst = []
        mse = tf.keras.losses.MeanSquaredError()
        initializer = True
        for epoch in range(pretraining_config.NUM_EPOCHS):

            optimizer.learning_rate = lr_schedule[epoch]
            optimizer.weight_decay = wd_schedule[epoch]

            for step, batch in enumerate(dataset):

                if initializer:
                    y = model(batch[0])
                    print(model.Encoder.summary())
                    print(model.Decoder.summary())
                    print(model.summary())
                    initializer = False

                with tf.GradientTape() as tape:
                    [_, output] = model(batch[0])

                    loss = mse(batch[0], output)
                    loss_lst.append(loss)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # test loss
            for step, batch in enumerate(dataset_test):
                batch_t = batch[0]
                [_, output] = model(batch_t)
                test_loss = mse(batch_t, output)
                test_loss_lst.append(test_loss)

            wandb.log({
                "Train Loss": np.mean(loss_lst[-dataset.cardinality().numpy():]),
                "Valid Loss": np.mean(test_loss_lst[-dataset_test.cardinality().numpy():])})

            print("Epoch: ", epoch + 1, ", Train loss: ", np.mean(loss_lst[-dataset.cardinality().numpy():]),
                  ", Test loss: ", np.mean(test_loss_lst[-dataset_test.cardinality().numpy():]))
            if pretraining_config.EARLY_STOPPING:
                if early_stopper.early_stop(np.mean(test_loss_lst[-dataset_test.cardinality().numpy():])):  # test_loss
                    break

        if save_weights:  # add numbering system if file already exists
            save_dir = check_filepath_naming(save_dir)
            wandb.log({"Actual save name": save_dir})
            model.save_weights(save_dir)
            save_pseudo = save_dir

    elif pretrain_method == "NNCLR":

        # Prepare dataset
        dataset = tf.data.Dataset.zip((dataset, dataset)).prefetch(buffer_size=tf.data.AUTOTUNE)

        nnclr = NNCLR(model_config=model_config,
                      pretraining_config=pretraining_config,
                      encoder=model.Encoder,
                      contrastive_augmenter=pretraining_config.CONTRASTIVE_AUGMENTER,
                      classification_augmenter=pretraining_config.CLASSIFICATION_AUGMENTER)
        nnclr.compile(contrastive_optimizer=tf.keras.optimizers.Adam(learning_rate=pretraining_config.
                                                                     LEARNING_RATE_NNCLR),
                      probe_optimizer=tf.keras.optimizers.Adam(learning_rate=pretraining_config.LEARNING_RATE_NNCLR))
        from wandb.keras import WandbMetricsLogger
        nnclr.fit(dataset,
                  epochs=pretraining_config.NUM_EPOCHS_NNCLR,
                  validation_data=dataset_test,
                  verbose=0,
                  callbacks=[WandbMetricsLogger()])

        if save_weights:  # add numbering system if file already exists

            if "Pretrain_" in save_dir:
                # Split the path at 'Pretrain_' and take the second part
                pseudo = tf.keras.Sequential([nnclr.encoder, nnclr.projection_head])
                for step, batch in enumerate(dataset):
                    batch_t = batch[0][0]
                    break
                pseudo.predict(batch_t)

                isolated_string = save_dir.split("Pretrain_")[0]
                save_pseudo = isolated_string + "Pretrain_" + "NNCLR_" + save_dir.split("Pretrain_")[1]

                save_pseudo = check_filepath_naming(save_pseudo)
                pseudo.save_weights(save_pseudo)
                wandb.log({"Actual save name": save_pseudo})

    return save_pseudo
