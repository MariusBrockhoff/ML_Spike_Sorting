# -*- coding: utf-8 -*-
#from models.AttnAE_1 import TransformerEncoder_AEDecoder, CustomSchedule
import tensorflow_addons as tfa



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
            lr = config.LEARNING_RATE*min(epoch,config.LR_WARMUP)/config.LR_WARMUP
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
