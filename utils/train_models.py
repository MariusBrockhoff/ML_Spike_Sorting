# -*- coding: utf-8 -*-
#from models.AttnAE_1 import TransformerEncoder_AEDecoder, CustomSchedule
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

def train_model(model, config, dataset, dataset_test, save_weights, save_dir):
    lr = config.LEARNING_RATE
    wd = config.WEIGHT_DECAY

    if config.WITH_WD:
        optimizer = tfa.optimizers.AdamW(weight_decay=wd, learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss_lst = []
    test_loss_lst = []

    mse = tf.keras.losses.MeanSquaredError()

    for epoch in range(config.NUM_EPOCHS):

        if config.WITH_WARMUP:
            lr = config.LEARNING_RATE*min(epoch, config.LR_WARMUP)/config.LR_WARMUP
            if epoch>config.LR_WARMUP:
                lr = config.LEARNING_RATE
                lr = config.LR_FINAL + .5*(lr-config.LR_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
                wd = wd + .5*(wd-config.WD_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
        else:
            lr = config.LR_FINAL + .5*(lr-config.LR_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))
            wd = wd + .5*(wd-config.WD_FINAL)*(1+np.cos(epoch*np.pi/config.NUM_EPOCHS))


        optimizer.learning_rate = lr
        optimizer.weight_decay = wd

        for step, batch in enumerate(dataset):
            batch_s = batch[0]
            with tf.GradientTape() as tape:
                [_, _, output] = model(batch_s)

                loss = mse(batch_s, output)
                grads = tape.gradient(loss,model.trainable_weights)
                optimizer.apply_gradients(zip(grads,model.trainable_weights))
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
