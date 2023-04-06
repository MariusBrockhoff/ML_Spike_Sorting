# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs)

    epoch_counter = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(epoch_counter * np.pi / epochs))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule


class SpikeAugmentation(tf.keras.Model):

    def __init__(self,

                 apply_noise=True,

                 max_noise_lvl=0.1,

                 apply_flip=True,

                 flip_probability=0.5,

                 apply_hshift=True,

                 max_hshift=None):

        super(SpikeAugmentation, self).__init__()

        self.apply_noise = apply_noise

        self.max_noise_lvl = max_noise_lvl

        self.apply_flip = apply_flip

        self.flip_probability = flip_probability

        self.apply_hshift = apply_hshift

        self.max_hshift = max_hshift

    def call(self, inputs):

        seq_len = inputs.shape[1]

        # noise

        if self.apply_noise:

            noise_lvl = np.random.uniform(0, self.max_noise_lvl)

            noise = tf.random.normal(shape=tf.shape(inputs), stddev=noise_lvl)

            inputs += noise

        #Horiz Trace Flipping

        if self.apply_flip:

            if np.random.rand() < self.flip_probability:

                inputs = np.flip(inputs, axis=1)

        # Horizontal shit

        if self.apply_hshift:

            if self.max_hshift==None:

                shift = np.random.randint(seq_len, size=1)

                inputs = np.roll(inputs, shift[0], axis=1)

            else:

                shift = np.random.randint(-self.max_hshift, self.max_hshift, size=1)

                inputs = np.roll(inputs, shift[0], axis=1)

        return inputs


def random_spike_subsetting(dat, cropping_percentage):

    n, seq_len = dat.shape

    red_seq_len = int(seq_len * cropping_percentage)

    start_ind = np.random.randint(seq_len - red_seq_len, size=1)

    end_ind = start_ind + red_seq_len

    cropped_spikes = dat[:, start_ind[0]:end_ind[0]]

    return tf.cast(cropped_spikes, dtype=tf.float32)


class DINOSpikeAugmentation(tf.keras.Model):

    def __init__(self,

                 max_noise_lvl=0.1,

                 crop_pcts=[0.8, 0.4],

                 number_local_crops=5):

        super(DINOSpikeAugmentation, self).__init__()

        self.max_noise_lvl = max_noise_lvl

        self.crop_pcts = crop_pcts

        self.number_local_crops = number_local_crops

    def call(self, inputs):

        # noise

        noise_lvl = np.random.uniform(0, self.max_noise_lvl)

        noise = tf.random.normal(shape=tf.shape(inputs), stddev=noise_lvl)

        inputs += noise

        # 0.5 flip the traces

        if np.random.rand() < 0.5:
            inputs = np.flip(inputs, axis=1)

        # Horizontal shit

        seq_len = inputs.shape[1]

        shift = np.random.randint(seq_len, size=1)

        inputs = np.roll(inputs, shift[0], axis=1)

        # Sub-setting for teacher and student

        augs = []

        for j in range(2):
            aug_t = random_spike_subsetting(inputs, self.crop_pcts[0])

            augs.append(aug_t)

        for i in range(self.number_local_crops):
            aug_s = random_spike_subsetting(inputs, self.crop_pcts[1])

            augs.append(aug_s)

        return augs[:2], augs

def train_model(model, config, dataset, dataset_test, save_weights, save_dir):

    if config.WITH_WARMUP:
        lr_schedule = cosine_scheduler(config.LEARNING_RATE, config.LR_FINAL, config.NUM_EPOCHS,
                                       warmup_epochs=config.LR_WARMUP, start_warmup_value=0)
    else:
        lr_schedule = cosine_scheduler(config.LEARNING_RATE, config.LR_FINAL, config.NUM_EPOCHS,
                                       warmup_epochs=0, start_warmup_value=0)
    wd_schedule = cosine_scheduler(config.WEIGHT_DECAY, config.WD_FINAL, config.NUM_EPOCHS,
                                   warmup_epochs=0, start_warmup_value=0)
    if config.WITH_WD:
        optimizer = tfa.optimizers.AdamW(weight_decay=wd_schedule[0], learning_rate=lr_schedule[0])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule[0])

    loss_lst = []
    test_loss_lst = []
    mse = tf.keras.losses.MeanSquaredError()

    augmenter = SpikeAugmentation(apply_noise=config.APPLY_NOISE, max_noise_lvl=config.MAX_NOISE_LVL,
                                  apply_flip=config.APPLY_FLIP, flip_probability=config.FLIP_PROBABILITY,
                                  apply_hshift=config.APPLY_HSHIFT, max_hshift=config.MAX_HSHIFT)

    for epoch in range(config.NUM_EPOCHS):

        optimizer.learning_rate = lr_schedule[epoch]
        optimizer.weight_decay = wd_schedule[epoch]

        for step, batch in enumerate(dataset):
            
            if config.DATA_AUG:
                batch_s = augmenter(batch[0])
            else:
                batch_s = batch[0]
                
            with tf.GradientTape() as tape:
                [_, _, output] = model(batch_s)

                loss = mse(batch_s, output)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
        loss_lst.append(loss)

        #test loss
        for step, batch in enumerate(dataset_test):
            batch_t = batch[0]
            [_, _, output] = model(batch_t)
            break
        test_loss = mse(batch_t, output)
        test_loss_lst.append(test_loss)

        print("Epoch: ", epoch+1, ", Train loss: ", loss, ", Test loss: ", test_loss)

    if save_weights:
        model.save_weights(save_dir)

    return loss_lst, test_loss_lst
