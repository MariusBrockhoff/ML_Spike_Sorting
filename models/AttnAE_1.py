import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment


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


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # masking zero values
    # print('shape of real:', real.shape)
    # print('shape of pred:', pred.shape)
    loss_ = loss_object(real, pred)
    # print('loss_:', loss_)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


# NOT USED AT THIS STAGE
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.float64))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float64)
    mask = tf.cast(mask, dtype=tf.float64)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


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
        arg1 = tf.math.rsqrt(float(step))
        arg2 = float(step) * (self.warmup_steps ** -1.5)

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


# AttnEncoder
class AttnEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, dropout=0.1):
        super(AttnEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = OneDtoTwoDLayer(seq_length=maximum_position_encoding, d_model=d_model)
        self.embedding = tf.keras.layers.Dense(d_model, name='dense_embedding')

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]  # shape of x: (batch_size, input_seq_len, d_model)

        # adding position encoding.

        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) #TODO: maybe insert here normalization step

        x = tf.expand_dims(x, -1)
        x = self.embedding(x)
        #print('After Embedding Shape: {}'.format(x.shape))

        x += self.pos_encoding
        #print('After Pos Encoding Shape: {}'.format(x.shape))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            #print('Encoder Layer {} Output Shape: {}'.format(i, x.shape))

        return x  # (batch_size, input_seq_len, d_model)


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


# TransformerEncoder + AEDecoder
class TransformerEncoder_AEDecoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, dropout, dec_dims, reg_value,
                 latent_len):
        super(TransformerEncoder_AEDecoder, self).__init__()

        self.encoder = AttnEncoder(num_layers, d_model, num_heads, dff, pe_input, dropout)

        self.reduce_pos_enc = tf.keras.layers.Dense(1, activation='relu', activity_regularizer=tf.keras.regularizers.l1(reg_value))
        self.reshape_pos_enc = tf.keras.layers.Reshape((pe_input,), input_shape=(pe_input, 1))

        self.latent_map = tf.keras.layers.Dense(latent_len, activation='relu',
                                                activity_regularizer=tf.keras.regularizers.l1(reg_value))

        self.decoder_layers = [
            tf.keras.layers.Dense(dec_dims[i], activation='relu', kernel_initializer='glorot_uniform',
                                  name='dense_decoder_%d' % i) for i in range(len(dec_dims))]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.final_dense = tf.keras.layers.Dense(pe_input)

        self.d_model = d_model

    def call(self, inp, training):
        # Keras models prefer if you pass all your inputs in the first argument

        #print('inp input', inp.shape)
        enc_output = self.encoder(inp, training, mask=None)  # (batch_size, inp_seq_len, d_model)
        #print('enc_output input', enc_output.shape)

        latent_vec = self.reduce_pos_enc(enc_output)
        latent_vec = tf.keras.layers.Reshape((latent_vec.shape[0], latent_vec.shape[1]), input_shape=(latent_vec.shape[0], latent_vec.shape[1], latent_vec.shape[2]))(latent_vec)
        # latent_vec = tf.math.reduce_mean(enc_output, axis=-1)
        #print('shape after reduce_pos_enc', latent_vec.shape)

        latent_vec = self.latent_map(latent_vec)
        #print('shape of latent_vec', latent_vec.shape)

        x = latent_vec

        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x)
            x = self.dropout(x, training=training)
            #print('shape after decoder layer:', x.shape)

        final_output = self.final_dense(x)
        #print('final_output input',final_output.shape)

        return inp, latent_vec, final_output
