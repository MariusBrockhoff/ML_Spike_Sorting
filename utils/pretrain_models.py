# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import wandb

from os import path
from utils.evaluation import *
#temp
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn import metrics


def check_filepath_naming(filepath):
    if path.exists(filepath):
        numb = 1
        while True:
            newPath = "{0}_{2}{1}".format(*path.splitext(filepath) + (numb,))
            if path.exists(newPath):
                numb += 1
            else:
                return newPath
    return filepath

def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs)

    epoch_counter = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(epoch_counter * np.pi / epochs))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule

class EarlyStopper:
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


class RandomResizedCrop1D(tf.keras.layers.Layer):
    def __init__(self, scale, seq_len):
        super().__init__()
        self.scale = scale
        self.seq_len = seq_len

    def build(self, input_shape):
        # This layer does not have any trainable weights, so we pass
        super().build(input_shape)

    def call(self, sequences):
        batch_size = tf.shape(sequences)[0]

        random_scale = tf.random.uniform((), self.scale[0], self.scale[1])
        new_length = tf.cast(tf.round(random_scale * tf.cast(self.seq_len, tf.float32)), tf.int32)

        max_start_idx = tf.maximum(0, self.seq_len - new_length + 1)
        start_idx = tf.random.uniform((), 0, max_start_idx, dtype=tf.int32)
        end_idx = start_idx + new_length

        cropped_sequences = sequences[:, start_idx:end_idx]


        # Resizing: temporarily swap batch and sequence dimensions
        cropped_sequences_swapped = tf.transpose(cropped_sequences, [1, 0])
        cropped_sequences_3d = tf.expand_dims(cropped_sequences_swapped, axis=-1)
        resized_sequences_3d = tf.image.resize(
            cropped_sequences_3d, [self.seq_len, batch_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        # Swap the dimensions back and remove the extra dimension to obtain the resized 1D sequences
        resized_sequences_swapped = tf.squeeze(resized_sequences_3d, axis=-1)
        resized_sequences = tf.transpose(resized_sequences_swapped, [1, 0])

        return resized_sequences

class RandomNoise(tf.keras.layers.Layer):
    def __init__(self, apply_noise=True, max_noise_lvl=1.0, seed=None):
        super().__init__()
        self.apply_noise = apply_noise
        self.max_noise_lvl = max_noise_lvl
        self.seed = seed

    def build(self, input_shape):
        # This layer does not have any trainable weights, so we pass
        super().build(input_shape)

    def call(self, inputs):
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

            inputs += noise  # Simpler addition operation

        return inputs


def spikeaugmentation(apply_noise, max_noise_lvl, scale, name):
    return tf.keras.Sequential(
        [
            #tf.keras.Input(shape=(63,)),
            RandomNoise(apply_noise=apply_noise, max_noise_lvl=max_noise_lvl),
            #RandomResizedCrop1D(scale=scale, seq_len=input_shape[0]),
        ],
        name=name,
    )

class ConditionalAugmentation(tf.keras.layers.Layer):
    def __init__(self, augmenter, **kwargs):
        super(ConditionalAugmentation, self).__init__(**kwargs)
        self.augmenter = augmenter

    def call(self, inputs, training=None):
        if training:
          return self.augmenter(inputs)
        return inputs


class NNCLR(tf.keras.Model):
    def __init__(self, model_config, pretraining_config, encoder, contrastive_augmenter, classification_augmenter):
        super().__init__()
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
            [tf.keras.layers.Input(shape=(self.projection_width,)), tf.keras.layers.Dense(pretraining_config.N_CLUSTERS)], name="linear_probe"
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

        batch_size = tf.shape(features_1, out_type=tf.float32)[0]
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






def pretrain_model(model, model_config, pretraining_config, pretrain_method, dataset, dataset_test, save_weights, save_dir):

    if pretrain_method == "reconstruction":
        if pretraining_config.EARLY_STOPPING:
            early_stopper = EarlyStopper(patience=pretraining_config.PATIENCE, min_delta=pretraining_config.MIN_DELTA,
                                         baseline=pretraining_config.BASELINE)

        if pretraining_config.WITH_WARMUP:
            lr_schedule = cosine_scheduler(pretraining_config.LEARNING_RATE, pretraining_config.LR_FINAL, pretraining_config.NUM_EPOCHS,
                                           warmup_epochs=pretraining_config.LR_WARMUP, start_warmup_value=0)
        else:
            lr_schedule = cosine_scheduler(pretraining_config.LEARNING_RATE, pretraining_config.LR_FINAL, pretraining_config.NUM_EPOCHS,
                                           warmup_epochs=0, start_warmup_value=0)
        wd_schedule = cosine_scheduler(pretraining_config.WEIGHT_DECAY, pretraining_config.WD_FINAL, pretraining_config.NUM_EPOCHS,
                                       warmup_epochs=0, start_warmup_value=0)
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


            #test loss
            for step, batch in enumerate(dataset_test):
                batch_t = batch[0]
                [_, output] = model(batch_t)
                test_loss = mse(batch_t, output)
                test_loss_lst.append(test_loss)

            wandb.log({
                "Train Loss": np.mean(loss_lst[-dataset.cardinality().numpy():]),
                "Valid Loss": np.mean(test_loss_lst[-dataset_test.cardinality().numpy():])})

            print("Epoch: ", epoch + 1, ", Train loss: ", np.mean(loss_lst[-dataset.cardinality().numpy():]), ", Test loss: ", np.mean(test_loss_lst[-dataset_test.cardinality().numpy():]))
            if pretraining_config.EARLY_STOPPING:
                if early_stopper.early_stop(np.mean(test_loss_lst[-dataset_test.cardinality().numpy():])): #test_loss
                    break


        if save_weights: #add numbering system if file already exists
            save_dir = check_filepath_naming(save_dir)
            wandb.log({"Actual save name": save_dir})
            model.save_weights(save_dir)



    elif pretrain_method == "NNCLR":

        #Prepare dataset
        dataset = tf.data.Dataset.zip((dataset, dataset)).prefetch(buffer_size=tf.data.AUTOTUNE)

        nnclr = NNCLR(model_config=model_config, pretraining_config=pretraining_config, encoder=model.Encoder,
                      contrastive_augmenter=pretraining_config.CONTRASTIVE_AUGMENTER,
                      classification_augmenter=pretraining_config.CLASSIFICATION_AUGMENTER)
        nnclr.compile(contrastive_optimizer=tf.keras.optimizers.Adam(learning_rate=pretraining_config.LEARNING_RATE_NNCLR),
                      probe_optimizer=tf.keras.optimizers.Adam(learning_rate=pretraining_config.LEARNING_RATE_NNCLR))
        from wandb.keras import WandbMetricsLogger
        nnclr.fit(dataset, epochs=pretraining_config.NUM_EPOCHS_NNCLR, validation_data=dataset_test, verbose=1, callbacks=[WandbMetricsLogger()])

        if save_weights: #add numbering system if file already exists

            if "Pretrain_" in save_dir:
                # Split the path at 'Pretrain_' and take the second part
                pseudo = tf.keras.Sequential([nnclr.encoder,
                                     nnclr.projection_head])
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



