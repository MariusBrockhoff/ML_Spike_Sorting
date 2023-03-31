# -*- coding: utf-8 -*-
#from models.AttnAE_1 import TransformerEncoder_AEDecoder, CustomSchedule
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


def train(batches_train, batches_test, loss_object, epochs, plot, save):

    if plot:
        plot_reconstruction(sample_batch=batches_test[1], index=0, method=method_plot_annotation, epochs=0,
                            parameters='')

    global train_loss_history, test_loss_history
    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        test_loss.reset_states()

        for (batch, inp) in enumerate(batches_train):

            tar_inp = inp[:, :-1]
            tar_real = inp[:, 1:]

            with tf.GradientTape() as tape:
                predictions, _, _ = transformer([inp, tar_inp], training=True)
                # print('predictions shape',predictions.shape)

                # predictions = np.delete(predictions1, -1, axis=1) #line used for only start codon (no stop codon)
                loss = loss_object(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)


        print(f'Epoch {epoch + 1} Train Loss: {train_loss.result():.4f}')  # Accuracy {train_accuracy.result():.4f}
        train_loss_history.append(train_loss.result())

        for (batch, inp) in enumerate(batches_test):
            tar_inp = inp[:, :-1]
            tar_real = inp[:, 1:]

            [input, latent, prediction] = model(batch_s)
            predictions, _, _, = transformer(inp, training=True)

            loss = loss_object(tar_real, predictions)

            test_loss(loss)

        print(f'Epoch {epoch + 1} Test Loss: {test_loss.result():.4f}')
        test_loss_history.append(test_loss.result())
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

        trainableParams = np.sum([np.prod(v.get_shape()) for v in transformer.trainable_weights])

        if (epoch + 1) % 10 == 0 and plot:
            plot_reconstruction(sample_batch=batches_test[1], index=0, method=method_plot_annotation,
                                epochs=(epoch + 1), parameters=trainableParams)

        if (epoch + 1) == epochs and save:
            transformer.save(
                f'saved_models/trials_model_epochs_{epochs}_{method_plot_annotation}_dataPrep_{data_prep}_dimsRed_{str(dims_red)}_numLayers_{num_layers}_dModel_{d_model}_dff_{dff}_numHeads_{num_heads}_dropout_{dropout_rate}')

"""
def train_transformer(batches_train, batches_test, signal_length, data_prep, num_layers, d_model, num_heads, dff, dropout_rate, dec_dims, epochs, plot):

    # data_prep =  'embedding', 'broadcasting', 'spectrogram' (string unqual to former two uses broadcasting as well)
    # num_layers = number of attention modules
    # d_model: embedding dimension --> ['gradient', 'perceiver-style', 'FT'] = [choose, 8, choose] #128
    # num_heads = (assert d_model//num_heads)
    # dff = shape of dense layer in attention module
    # dropout
    # dec_dims: list of dimensions of dense decoder
    # epochs: int specify number of epochs to train
    # plot: boolean plot reconstruction after every 10 epochs

    global learning_rate, optimizer, loss_object, train_loss, test_loss, transformer

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9) #learning_rate

    loss_object = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    transformer = TransformerEncoder_AEDecoder(
        data_prep=data_prep,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        pe_input=signal_length,
        dropout=dropout_rate,
        dec_dims=dec_dims)

    train(batches_train=batches_train, batches_test=batches_test, loss_object=loss_object, epochs=epochs, plot=plot,
          save=False)
"""


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

    for epoch in range(config.NUM_EPOCHS):

        optimizer.learning_rate = lr_schedule[epoch]
        optimizer.weight_decay = wd_schedule[epoch]

        for step, batch in enumerate(dataset):
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
