import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import os
import sys

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input, TimeDistributed, Dropout
from keras.models import Model
from keras import callbacks
from keras.initializers import VarianceScaling
from keras.constraints import UnitNorm, Constraint
from scipy.fft import fft

import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


###auxilary functions for transformer build-up: positional encoding, masks, scaled_dot_product, point-wise feed forwards layer
###                   for data preprocessing: gradient_transform, make_batches
###                   for data visualization: reconstruction, spectrogram
###                   for training: train, eval, loss, acc

# positional encoding
def get_angles(pos, i, d_model):
    # args:  - pos: is the position of the input embedding in the sequence (hello world, how are you: pos(world)=2)
    #       - i: is the dimension of the input in the input embedding ("world" --> [0.1,0.4,0.2,0.3]: i=2 refers to input 0.4 of "world" embedding)
    #       - d_model: dimension of the embedding ("world" --> [0.1,0.4,0.2,0.3]: d_model = 4)
    #       --> in our setting: no embedding necessary, so always i=1 and d_model=1
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# Masking for padding: not sure if we need this / do we have padding
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# Masking future tokens in a sequence: used in decoder inputs
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# only mask current input
# version 1: upper off-diagonal
def create_current_input_mask_off_diag(size):
    off_diag = tf.linalg.band_part(tf.ones((size, size)), -2, 1) - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return off_diag  # (seq_len, seq_len)


# version 2: diagonal
def create_current_input_mask_diag(size):
    diag = tf.linalg.band_part(tf.ones((size, size)), 0, 0)
    return diag  # (seq_len, seq_len)


# version3: full blocking mask
def create_blocking_mask(size):
    block = tf.ones((size, size))
    return block  # (seq_len, seq_len)


# scaled dot product for attention layers
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', name='pointwise_ffn_dense1'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, name='pointwise_ffn_dense2')  # (batch_size, seq_len, d_model)
    ])


def gradient_transform(X, sr=24000):
    number_sampling_points = X.shape[1]
    time_step = 1 / sr * 10 ** 6  # in us
    X_grad = np.empty(shape=(X.shape[0], X.shape[1] - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - 1):
            X_grad[i, j] = (X[i, j + 1] - X[i, j]) / time_step

    return X_grad


def get_spectrogram(waveform, d_model):
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=2 * d_model,
        frame_step=1)  # frame_length=32, frame_step=1: (64,1) sequence transformed into (33,17) spectrogram --> use d_model=16
    #             16             1                                   (49,9)                               8
    #              8             1                                   (57,5)                               4
    spectrogram = tf.abs(spectrogram)

    signal_length = spectrogram.shape[-2]

    # neglect bin with highest frequencises to obtain a d_model of power of 2
    spectrogram = spectrogram[:, :, :-1]  # shape: (samples, seq_length, d_model)

    return spectrogram, signal_length


def plot_spectrogram(spectrogram, ax):
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(np.array(spectrogram).T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    im = ax.pcolormesh(X, Y, log_spec)
    fig.colorbar(im, ax=ax)


def make_batches(data, batch_size, shuffle, preprocessing_method):
    if shuffle:
        data = tf.random.shuffle(data)

    mod = data.shape[0] % batch_size

    if data.shape[0] % batch_size != 0:
        data = data[:-mod]

    batch_number = int(data.shape[0] / batch_size)

    if preprocessing_method != 0:
        data = tf.reshape(data, shape=(batch_number, batch_size, data.shape[-2], data.shape[-1]))

    else:
        data = tf.reshape(data, shape=(batch_number, batch_size, data.shape[-1]))

    return data


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # masking zero values
    # print('shape of real:', real.shape)
    # print('shape of pred:', pred.shape)
    loss_ = loss_object(real, pred)
    # print('loss_:', loss_)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _, _ = transformer([inp, tar_inp],
                                        training=True)
        # print('predictions shape',predictions.shape)

        # predictions = np.delete(predictions1, -1, axis=1) #line used for only start codon (no stop codon)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)


def ae_train_step(inp):
    with tf.GradientTape() as tape:
        predictions, _ = ae(inp, training=True)
        # print('predictions shape',predictions.shape)

        loss = loss_function(inp, predictions)

    gradients = tape.gradient(loss, ae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ae.trainable_variables))

    train_loss(loss)


def eval_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _, _, = transformer([inp, tar_inp],
                                     training=True)

    loss = loss_function(tar_real, predictions)

    test_loss(loss)


# NOT USED AT THIS STAGE
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.float64))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float64)
    mask = tf.cast(mask, dtype=tf.float64)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train(epochs, plot, save):
    EPOCHS = epochs

    if plot:
        plot_reconstruction(sample_batch=batches_test[1], index=0, method=method_plot_annotation, epochs=0,
                            parameters='')

    global train_loss_history, test_loss_history
    train_loss_history = []
    test_loss_history = []

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        test_loss.reset_states()

        for (batch, inp) in enumerate(batches_train):
            train_step(inp, inp)
            # if batch % 50 == 0:
            #  print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}') #Accuracy {train_accuracy.result():.4f}

        # if (epoch + 1) % 5 == 0:
        #  ckpt_save_path = ckpt_manager.save()
        #  print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Train Loss: {train_loss.result():.4f}')  # Accuracy {train_accuracy.result():.4f}
        train_loss_history.append(train_loss.result())

        for (batch, inp) in enumerate(batches_test):
            eval_step(inp, inp)

        print(f'Epoch {epoch + 1} Test Loss: {test_loss.result():.4f}')
        test_loss_history.append(test_loss.result())
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

        trainableParams = np.sum([np.prod(v.get_shape()) for v in transformer.trainable_weights])

        if (epoch + 1) % 10 == 0 and plot:
            plot_reconstruction(sample_batch=batches_test[1], index=0, method=method_plot_annotation,
                                epochs=(epoch + 1), parameters=trainableParams)

        if (epoch + 1) == EPOCHS and save:
            transformer.save(
                f'saved_models/trials_model_epochs_{epochs}_{method_plot_annotation}_dataPrep_{data_prep}_dimsRed_{str(dims_red)}_numLayers_{num_layers}_dModel_{d_model}_dff_{dff}_numHeads_{num_heads}_dropout_{dropout_rate}')


def train_ae(epochs, plot, save):
    EPOCHS = epochs

    if plot:
        # print('shape of batch to plot:', batches_train.shape)
        plot_reconstruction(sample_batch=batches_train[1], index=0, method=method_plot_annotation, epochs=0,
                            parameters='')

    global train_loss_history, test_loss_history
    train_loss_history = []
    test_loss_history = []

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        # test_loss.reset_states()

        for (batch, inp) in enumerate(batches_train):
            ae_train_step(inp)

        print(f'Epoch {epoch + 1} Train Loss: {train_loss.result():.4f}')  # Accuracy {train_accuracy.result():.4f}
        train_loss_history.append(train_loss.result())

        trainableParams = np.sum([np.prod(v.get_shape()) for v in ae.trainable_weights])

        if (epoch + 1) % 10 == 0 and plot:
            plot_reconstruction(sample_batch=batches_train[1], index=0, method=method_plot_annotation,
                                epochs=(epoch + 1), parameters=trainableParams)


# inference
def concatenate2D1D(vec):
    # 1040 dims for num_heads=4, d_model=16; 520 dims for num_heads=4, d_model=8
    print('dim:', vec.shape[0] * vec.shape[1])
    return tf.reshape(vec, shape=(vec.shape[0] * vec.shape[1]))


def predict_and_latent(data, architecture):
    predict_vec, latent_vec, _ = transformer([data, data[:, :-1]], training=False)

    if architecture == 'untouched':
        latent_vec = concatenate2D1D(latent_vec)

    return predict_vec, latent_vec


def plot_reconstruction(sample_batch, index, method, epochs, parameters):
    predict_batch, _, _ = transformer([sample_batch, sample_batch[:, :-1]], training=False)
    # predict_batch, _ = ae(sample_batch, training=False) #for training autoencoder
    # print('shape of prediction:', predict_batch.shape)

    if method == 'perceiver-style':
        xlabel = 'potential'
        input = tf.reshape(sample_batch[index], (sample_batch[index].shape[0] * sample_batch[index].shape[1]))[8:-8]
        print('shape of prediction', predict_batch[index].shape)
        reconstruction = tf.reshape(predict_batch[index],
                                    (predict_batch[index].shape[0] * predict_batch[index].shape[1]))[:-8]
        print('shape of reconstruction', reconstruction.shape)

    elif method == 'gradient' or method == 'FT':
        xlabel = method
        input = sample_batch[index][1:-1]
        # input=sample_batch[index][1:-1,1]
        reconstruction = predict_batch[index][:-1]
        # reconstruction=predict_batch[index][1:-1,1] #for training autoencoder

    if not method == 'spectrogram':
        plt.plot(input, label='input')
        plt.plot(reconstruction, label='reconstruction')
        plt.legend()
        plt.xlabel('timesteps')
        plt.ylabel(xlabel)
        plt.title('reconstruction: {} after {} epochs'.format(method,
                                                              epochs))  # epochs (model: {} parameters)'.format(method, epochs, parameters))
        plt.show()

    elif method == 'spectrogram':
        xlabel = method
        input = sample_batch[index][1:, :]
        reconstruction = predict_batch[index]

        global fig
        fig, axes = plt.subplots(2, figsize=(12, 8))
        # plot_spectrogram(input, axes[0])

        log_spec = np.log(np.array(input).T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(input), num=width, dtype=int)
        Y = range(height)
        im = axes[0].pcolormesh(X, Y, log_spec)

        axes[0].set_title('input: spectrogram')

        # plot_spectrogram(reconstruction, axes[1])

        log_spec2 = np.log(np.array(reconstruction).T + np.finfo(float).eps)
        height2 = log_spec2.shape[0]
        width2 = log_spec2.shape[1]
        X2 = np.linspace(0, np.size(input), num=width2, dtype=int)
        Y2 = range(height2)
        im2 = axes[1].pcolormesh(X2, Y2, log_spec2)

        axes[1].set_title(
            'reconstruction: {} after {} epochs (model: {} parameters)'.format(method, epochs, parameters))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


# classes for  transformer: attention, encoder layer, encoder, decoder layer, decoder, transformer
#             bottlenecks: autoencoder, OneDtoTwoDLayer

# scheduler
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# multiHeadAttention
class multiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(multiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name='dense_wq')
        self.wk = tf.keras.layers.Dense(d_model, name='dense_wk')
        self.wv = tf.keras.layers.Dense(d_model, name='dense_wv')

        self.dense = tf.keras.layers.Dense(d_model, name='dense_last')

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# EncoderLayer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = multiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm2')

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# DecoderLayer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = multiHeadAttention(d_model, num_heads)
        self.mha2 = multiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = OneDtoTwoDLayer(seq_length=maximum_position_encoding, d_model=d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask, data_prep):

        seq_len = tf.shape(x)[1]  # shape of x: (batch_size, input_seq_len, d_model)

        # adding position encoding.

        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) #TODO: maybe insert here normalization step

        if data_prep == 'gradient' or data_prep == 'FT':
            x = tf.expand_dims(x, axis=-1)

        elif data_prep == 'embedding':
            x = self.embedding(x)

        x += self.pos_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = OneDtoTwoDLayer(seq_length=maximum_position_encoding - 1, d_model=d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask, data_prep):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  #TODO: maybe insert here normalization step
        if data_prep == 'gradient' or data_prep == 'FT':
            x = tf.expand_dims(x, axis=-1)

        elif data_prep == 'embedding':
            x = self.embedding(x)

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        return x, attention_weights  # shape of x (batch_size, target_seq_len, d_model)


# Autoencoder
class autoencoder(tf.keras.layers.Layer):
    """
    Fully connected auto-encoder model with 2Dto1D and 1Dto2D first and last layer, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is first dense layer dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        dropout: dropout probabilty
        d_model: embedding dimension of sequence
        signal_length: length of sequence
        act: activation, not applied to Input, Hidden and Output layers
    return:
        x' = f(g(x)) with x=input; g() encoder mapping; f() decoder mapping
        latent vector
    """

    def __init__(self, dims, dropout, d_model, signal_length, act='relu', init='glorot_uniform'):
        super(autoencoder, self).__init__()

        self.dims = dims
        self.act = act
        self.init = init
        self.signal_length = signal_length

        self.encoder_layers = [tf.keras.layers.Dense(self.dims[i], activation=self.act, kernel_initializer=self.init,
                                                     name='ae_encoder_%d' % i) for i in range(len(self.dims[:-1]))]
        self.hidden = tf.keras.layers.Dense(self.dims[-1], kernel_initializer=self.init,
                                            name='ae_encoder_%d' % (len(self.dims) - 1))
        self.decoder_layers = [tf.keras.layers.Dense(self.dims[i], activation=self.act, kernel_initializer=self.init,
                                                     name='ae_decoder_%d' % i) for i in
                               range(len(self.dims[:-1]) - 1, -1, -1)]
        self.last_dense_layer = tf.keras.layers.Dense(self.signal_length, kernel_initializer=self.init,
                                                      name='ae_decoder_last1D')

        self.exp_layer = OneDtoTwoDLayer(seq_length=signal_length, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):

        # 2Dto1D operation
        # x = tf.math.reduce_max(x, axis=-1)
        x = tf.math.reduce_mean(x, axis=-1)
        # print('shape after 2Dto1D:', x.shape)

        # internal layers in ae_encoder
        for i in range(len(self.dims[:-1])):
            x = self.encoder_layers[i](x)
            x = self.dropout(x, training=training)
            # print('shape after encoder layer:', x.shape)

        # hidden layer
        latent_vec = self.hidden(x)  # hidden layer, features are extracted from here
        # print('shape after latent_vec:', latent_vec.shape)
        x = latent_vec

        # internal layers in ae_decoder
        # for i in range(len(self.dims[:-1])-1, -1, -1):
        for i in range(len(self.dims[:-1])):
            x = self.decoder_layers[i](x)
            x = self.dropout(x, training=training)
            # print('shape after decoder layer:', x.shape)

        # back to default signal_length
        x = self.last_dense_layer(x)
        # print('shape after last decoder layer:', x.shape)

        # 1Dto2D layer to ensure shape(input2D) = shape(output2D)
        x = self.exp_layer(x)
        # print('shape after 1Dto2D:', x.shape)

        return x, latent_vec


# OneDtoTwoDLayer
class OneDtoTwoDLayer(tf.keras.layers.Layer):
    # input shape: (batch_size, seq_length)
    # output shape: (batch_size, seq_length, d_model)
    def __init__(self, seq_length, d_model, dropout=0.1, act='relu', init='glorot_uniform'):
        super(OneDtoTwoDLayer, self).__init__()

        self.seq_length = seq_length

        self.dropout = tf.keras.layers.Dropout(dropout)

        # same layer for every projection (scalar --> vector)
        # self.dense = tf.keras.layers.Dense(d_model, activation=act, kernel_initializer=init, name='dense_1Dto2D')

        # individual layer for every projection (scalar --> vector)
        self.dense_list = [
            tf.keras.layers.Dense(d_model, activation=act, kernel_initializer=init, name='dense_1Dto2D_%i' % i) for i in
            range(self.seq_length)]

    def call(self, x, training):
        scalars = tf.split(x, num_or_size_splits=self.seq_length, axis=-1)

        output_list = []
        for i in range(self.seq_length):
            out = self.dense_list[i](scalars[i])
            out = self.dropout(out, training=training)
            output_list.append(out)

        output = tf.convert_to_tensor(output_list)  # shape: (seq_length, batch_size, d_model)
        output = tf.transpose(output, perm=[1, 0, 2])  # shape: (batch_size, seq_length, d_model)

        return output


# Transformer
class Transformer(tf.keras.Model):
    def __init__(self, mask_method, data_prep, dims_red, num_layers, d_model, num_heads, dff, pe_input, pe_target,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, dropout)

        if len(dims_red) > 1:
            self.ae = autoencoder(dims_red, dropout, d_model, pe_input, act='relu', init='glorot_uniform')

        if len(dims_red) == 1:
            self.dim_exp_layer = OneDtoTwoDLayer(seq_length=pe_input, d_model=d_model)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, pe_target, dropout)

        if data_prep == 'perceiver-style' or data_prep == 'spectrogram':
            self.final_layer = tf.keras.layers.Dense(d_model)

        else:
            self.final_layer = tf.keras.layers.Dense(1)

        self.mask_method = mask_method
        self.data_prep = data_prep
        self.dims_red = dims_red
        self.signal_length = pe_input
        self.d_model = d_model

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs  # inp = input, tar = target

        look_ahead_mask = self.create_mask(tar, self.mask_method)

        enc_output = self.encoder(inp, training, mask=None,
                                  data_prep=self.data_prep)  # (batch_size, inp_seq_len, d_model)

        latent_vec = enc_output

        if len(self.dims_red) > 1:
            enc_output, latent_vec = self.ae(latent_vec)

        elif len(self.dims_red) == 1:
            # print('shape of 2Dto1D operation input:', enc_output.shape)
            # latent_vec = tf.math.reduce_max(latent_vec, axis=-1)
            latent_vec = tf.math.reduce_mean(latent_vec, axis=-1)
            # print('shape of 2Dto1D operation output:', latent_vec.shape)
            enc_output = self.dim_exp_layer(latent_vec)
            # print('shape of embedding output:', enc_output.shape)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, padding_mask=None, data_prep=self.data_prep)
        # print('shape of dec_output', dec_output.shape)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # print('shape of final_output', final_output.shape)

        final_output = tf.squeeze(final_output)
        # print('shape of final_output', final_output.shape)

        return final_output, latent_vec, attention_weights

    def create_mask(self, tar, mask_method):
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask tokens in the input according to mask method received by the decoder.
        look_ahead_mask = mask_method(tf.shape(tar)[1])

        return look_ahead_mask


# TransformerEncoder + AEDecoder
class TransformerEncoder_AEDecoder(tf.keras.Model):
    def __init__(self, data_prep, num_layers, d_model, num_heads, dff, pe_input, pe_target, dropout, dec_dims):
        super(TransformerEncoder_AEDecoder, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, dropout)

        self.decoder_layers = [
            tf.keras.layers.Dense(dec_dims[i], activation='relu', kernel_initializer='glorot_uniform',
                                  name='dense_decoder_%d' % i) for i in range(len(dec_dims))]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.final_dense = tf.keras.layers.Dense(pe_input - 1)
        self.twoD_expansion = OneDtoTwoDLayer(seq_length=pe_input - 1, d_model=d_model)  # if input is 2D

        self.data_prep = data_prep
        self.d_model = d_model

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs  # inp = input, tar = target
        # print('shape input',inp.shape)

        enc_output = self.encoder(inp, training, mask=None,
                                  data_prep=self.data_prep)  # (batch_size, inp_seq_len, d_model)
        # print('enc_output input',enc_output.shape)

        latent_vec = tf.math.reduce_mean(enc_output, axis=-1)
        # print('latent_vec input',latent_vec.shape)

        x = latent_vec

        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x)
            x = self.dropout(x, training=training)
            # print('shape after decoder layer:', x.shape)

        final_output = self.final_dense(x)
        # print('final_output input',final_output.shape)

        if self.d_model in [4, 8, 16]:
            final_output = self.twoD_expansion(final_output)

        return final_output, latent_vec, inp