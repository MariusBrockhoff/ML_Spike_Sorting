# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import wandb

from os import path
#TODO: add self-supervised backbone training


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

class DINOLoss(tf.keras.layers.Layer):
    def __init__(self, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp, ncrops,
                 center, center_momentum):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.center = center
        self.center_momentum = center_momentum
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = tf.reduce_mean(teacher_output, axis=0, keepdims=True)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def call(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = tf.split(student_out, num_or_size_splits=self.ncrops, axis=0)

        # teacher centering and sharpening
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = tf.nn.softmax((teacher_output - self.center) / teacher_temp, axis=-1)
        teacher_out = tf.split(teacher_out, num_or_size_splits=2, axis=0)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = tf.reduce_sum(-q * tf.nn.log_softmax(student_out[v], axis=-1), axis=-1)
                total_loss += tf.reduce_mean(loss)
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

def pretrain_model(model, config, pretrain_method, dataset, dataset_test, save_weights, save_dir):

    if config.EARLY_STOPPING:
        early_stopper = EarlyStopper(patience=config.PATIENCE, min_delta=config.MIN_DELTA, baseline=config.BASELINE)

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

    if config.DATA_AUG:
        augmenter = SpikeAugmentation(apply_noise=config.APPLY_NOISE, max_noise_lvl=config.MAX_NOISE_LVL,
                                      apply_flip=config.APPLY_FLIP, flip_probability=config.FLIP_PROBABILITY,
                                      apply_hshift=config.APPLY_HSHIFT, max_hshift=config.MAX_HSHIFT)


    if pretrain_method == "reconstruction":
        initializer = True
        for epoch in range(config.NUM_EPOCHS):

            optimizer.learning_rate = lr_schedule[epoch]
            optimizer.weight_decay = wd_schedule[epoch]

            for step, batch in enumerate(dataset):

                if config.DATA_AUG:
                    batch_s = augmenter(batch[0])
                else:
                    batch_s = batch[0]


                if initializer:
                    y = model(batch_s)
                    #model.Encoder.build((None, batch_s.shape[1]))
                    print(model.Encoder.summary())
                    #out_layer = model.Encoder.layers[-1]
                    #name = out_layer.name
                    #input_shape = out_layer.input_shape
                    #output_shape = out_layer.output_shape
                    #print('%s   input shape: %s, output_shape: %s.\n' % (name, input_shape, output_shape))
                    #model.Decoder.build(output_shape)
                    print(model.Decoder.summary())
                    #y = model(batch_s)
                    print(model.summary())
                    initializer = False


                with tf.GradientTape() as tape:
                    [_, output] = model(batch_s)

                    loss = mse(batch_s, output)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
            loss_lst.append(loss)

            #test loss
            for step, batch in enumerate(dataset_test):
                batch_t = batch[0]
                [_, output] = model(batch_t)
                test_loss = mse(batch_t, output)
                test_loss_lst.append(test_loss)

            wandb.log({
                "Epoch": epoch,
                "Train Loss": loss.numpy(),
                "Valid Loss": test_loss.numpy()})

            print("Epoch: ", epoch + 1, ", Train loss: ", loss.numpy(), ", Test loss: ", test_loss.numpy())

            if config.EARLY_STOPPING:
                if early_stopper.early_stop(np.mean(test_loss_lst[-10:])): #test_loss
                    break


        if save_weights: #add numbering system if file already exists
            save_dir = check_filepath_naming(save_dir)
            print("Corrected save name:", save_dir)
            model.save_weights(save_dir)

        return loss_lst, test_loss_lst, epoch+1


    elif pretrain_method == "NNCLR":
        #TODO: implement NNCLR pretraining
        print("NNCLR still to be implemented")

    elif pretrain_method == "DINO":

        loss_lst = []
        m_student = model[0]
        m_teacher = model[1]
        c = tf.zeros((1, config.LATENT_LEN))
        m = config.CENTERING_RATE
        l = config.LEARNING_MOMENTUM_RATE
        tau_student = config.STUDENT_TEMPERATURE
        tau_teacher = config.TEACHER_TEMPERATURE
        tau_teacher_final = config.TEACHER_TEMPERATURE_FINAL
        warmup = config.TEACHER_WARMUP

        dino_loss = DINOLoss(tau_teacher, tau_teacher_final, warmup, config.NUM_EPOCHS, tau_student,
                             config.NUMBER_LOCAL_CROPS + 2, c, m)

        augmenter = DINOSpikeAugmentation(max_noise_lvl=config.MAX_NOISE_LVL,
                                          crop_pcts=config.CROP_PCTS,
                                          number_local_crops=config.NUMBER_LOCAL_CROPS)
        for epoch in range(config.NUM_EPOCHS):

            optimizer.learning_rate = lr_schedule[epoch]
            optimizer.weight_decay = wd_schedule[epoch]
            l = config.LEARNING_MOMENTUM_RATE + .002 * np.cos(epoch * np.pi / config.NUM_EPOCHS)

            for step, batch in enumerate(dataset):

                if config.DATA_AUG:
                    batch_t, batch_s = augmenter(batch[0])

                with tf.GradientTape() as tape:
                    student_logs = []
                    for single_batch in batch_s:
                        _, single_student_logits = m_student(single_batch)
                        student_logs.append(single_student_logits)
                    student_logits = tf.concat(student_logs, axis=0)

                    with tape.stop_recording():
                        teacher_logs = []
                        for single_batch in batch_t:
                            _, single_teacher_logits = m_teacher(single_batch)
                            teacher_logs.append(single_teacher_logits)
                        teacher_logits = tf.concat(teacher_logs, axis=0)

                    loss = dino_loss(student_logits, teacher_logits, epoch)

                grads = tape.gradient(loss, m_student.trainable_weights)
                optimizer.apply_gradients(zip(grads, m_student.trainable_weights))
                loss_lst.append(loss)

            for layer_s, layer_t in zip(m_student.layers, m_teacher.layers):
                weights_s = layer_s.get_weights()
                weights_t = layer_t.get_weights()
                weights_update = []
                # Weights of the teacher are updated according to DINO idea
                for weight_s, weight_t in zip(weights_s, weights_t):
                    weights_update.append(
                        l * weight_t + (1 - l) * weight_s)

                layer_t.set_weights(weights_update)

            wandb.log({
                "Epoch": epoch,
                "DINO Loss": loss.numpy()})

            # if config.EARLY_STOPPING:
            # if early_stopper.early_stop(test_loss):
            # break

        if save_weights:
            save_dir = check_filepath_naming(save_dir)
            model.save_weights(save_dir)

        return loss_lst, loss_lst, epoch + 1

    else:
        print("Choose valid pre train method pls")