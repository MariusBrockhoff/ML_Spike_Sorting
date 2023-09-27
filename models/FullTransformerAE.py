# -*- coding: utf-8 -*-
"""
Definition of the AttnAE_2 Class for Spike Sorting
"""
import tensorflow as tf
import numpy as np
import math


from einops import rearrange, repeat
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as KL


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

def get_angles(pos, i, d_model):
  #args:  - pos: is the position of the input embedding in the sequence (hello world, how are you: pos(world)=2)
  #       - i: is the dimension of the input in the input embedding ("world" --> [0.1,0.4,0.2,0.3]: i=2 refers to input 0.4 of "world" embedding)
  #       - d_model: dimension of the embedding ("world" --> [0.1,0.4,0.2,0.3]: d_model = 4)
  #       --> in our setting: no embedding necessary, so always i=1 and d_model=1
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def fourier_encode(x, max_freq, num_bands=4, base=2):

    x = tf.expand_dims(x, -1)

    x = tf.cast(x, dtype=tf.float32)

    orig_x = x

    scales = tf.experimental.numpy.logspace(

        1.0,

        math.log(max_freq / 2) / math.log(base),

        num=num_bands,

        base=base,

        dtype=tf.float32,

    )

    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]



    x = x * scales * math.pi

    x = tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)

    x = tf.concat((x, orig_x), axis=-1)

    return x


class fourier_encoding(tf.keras.layers.Layer):
    # input shape: (batch_size, seq_lenght, n_features, 1)
    # output shape: (batch_size, seq_length*n_features, d_model)
    def __init__(self, max_freq, num_bands, base):
        super(fourier_encoding, self).__init__()

        self.max_freq = max_freq

        self.num_bands = num_bands

        self.base = base

    def fourier_encode(self, x, max_freq, num_bands, base):
        x = tf.expand_dims(x, -1)

        x = tf.cast(x, dtype=tf.float32)

        orig_x = x

        scales = tf.experimental.numpy.logspace(

            1.0,

            math.log(max_freq / 2) / math.log(base),

            num=num_bands,

            base=base,

            dtype=tf.float32,

        )

        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * math.pi

        x = tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)

        x = tf.concat((x, orig_x), axis=-1)

        return x

    def call(self, input):
        b, *axis, _ = input.shape

        axis_pos = list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis))

        pos = tf.stack(tf.meshgrid(*axis_pos, indexing="ij"), axis=-1)

        enc_pos = self.fourier_encode(pos,

                                      self.max_freq,

                                      self.num_bands,

                                      self.base)

        enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")

        enc_pos = rearrange(enc_pos, "... c -> (...) c")

        enc_pos = repeat(enc_pos, "... -> b ...", b=b)

        input = rearrange(input, "b ... d -> b (...) d")

        output = input + enc_pos

        return output


class SelfAttention(tf.keras.Model):

    def __init__(self,

                 embedding_dim=65,

                 dff=128,

                 attn_dim=64,

                 attn_heads=8,

                 dropout_rate=0.0):

        super(SelfAttention, self).__init__()

        self.Embedding_dim = embedding_dim

        self.dff = dff

        self.attn_dim = attn_dim

        self.attn_heads = attn_heads

        self.dropout_rate = dropout_rate

        self.attn = KL.MultiHeadAttention(self.attn_heads, self.attn_dim)

        self.attn_sum1 = KL.Add()

        self.attn_norm1 = KL.LayerNormalization(axis=-1)

        self.fin_norm = KL.LayerNormalization(axis=-1)

        self.attn_mlp = Sequential([KL.LayerNormalization(axis=-1),

                                    KL.Dense(self.dff, activation=tf.keras.activations.gelu),

                                    KL.Dense(self.Embedding_dim),

                                    KL.Dropout(self.dropout_rate)])

    def call(self, inputs):

        x = inputs

        normed_inputs = self.attn_norm1(inputs)

        x += self.attn(normed_inputs, normed_inputs, return_attention_scores=False)

        x += self.attn_mlp(x)

        x = self.fin_norm(x)

        return x


class AttnModule(tf.keras.layers.Layer):

    def __init__(self,

                 embedding_dim=65,

                 dff=128,

                 depth=8,

                 attn_dim=64,

                 attn_heads=8,

                 dropout_rate=0.2):
        super(AttnModule, self).__init__()

        self.Embedding_dim = embedding_dim

        self.dff = dff

        self.depth = depth

        self.attn_dim = attn_dim

        self.attn_heads = attn_heads

        self.dropout_rate = dropout_rate

        self.layers = [SelfAttention(d_model=self.Embedding_dim,

                                  dff=self.dff,

                                  attn_dim=self.attn_dim,

                                  attn_heads=self.attn_heads,

                                  dropout_rate=self.dropout_rate) for _ in range(self.depth)]

    def call(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)

        return x


class Encoder(tf.keras.Model):
    def __init__(self,
                 embedding_dim,
                 dff,
                 seq_len,
                 latent_len,
                 enc_depth,
                 enc_attn_dim,
                 enc_attn_heads,
                 enc_dropout_rate):
        super(Encoder, self).__init__()

        self.Embedding_dim = embedding_dim
        self.dff = dff
        self.seq_len = seq_len
        self.latent_len = latent_len
        self.ENC_depth = enc_depth
        self.ENC_attn_dim = enc_attn_dim
        self.ENC_attn_heads = enc_attn_heads
        self.ENC_dropout_rate = enc_dropout_rate

        self.embedding_enc = KL.Dense(self.Embedding_dim)  # Add normalization (*sqrt(embedding_dim))

        self.positional_enc = positional_encoding(self.seq_len, self.Embedding_dim)

        self.Encoder = AttnModule(self.Embedding_dim, self.dff, self.ENC_depth, self.ENC_attn_dim, self.ENC_attn_heads,
                                   self.ENC_dropout_rate)

        self.reduc_pos_enc = Sequential([KL.Dense(1),                               #KL.Dense(1, activation='relu'),
                                         KL.Reshape((self.seq_len,), input_shape=(self.seq_len, 1))])

        self.ENC_to_logits = KL.Dense(self.latent_len) # PREVIOUS: self.ENC_to_logits = Sequential([KL.Dense(self.latent_len, activation='relu',
        # activity_regularizer=tf.keras.regularizers.l1(self.reg_value))])

    def call(self, inputs):

        inputs = rearrange(inputs, "a b -> a b 1")
        inputs = self.embedding_enc(inputs)
        inputs += self.positional_enc

        # Encoder
        encoded = self.Encoder(inputs)
        encoded = self.reduc_pos_enc(encoded)
        latents = self.ENC_to_logits(encoded)

        return latents


class Decoder(tf.keras.Model):
    def __init__(self,
                 embedding_dim,
                 dff,
                 seq_len,
                 latent_len,
                 dec_depth,
                 dec_attn_dim,
                 dec_attn_heads,
                 dec_dropout_rate):
        super(Decoder, self).__init__()

        self.Embedding_dim = embedding_dim
        self.dff = dff
        self.seq_len = seq_len
        self.latent_len = latent_len
        self.DEC_depth = dec_depth
        self.DEC_attn_dim = dec_attn_dim
        self.DEC_attn_heads = dec_attn_heads
        self.DEC_dropout_rate = dec_dropout_rate

        self.embedding_dec = KL.Dense(self.Embedding_dim)

        self.positional_enc_dec = positional_encoding(self.latent_len, self.Embedding_dim)

        self.decoder = AttnModule(self.Embedding_dim, self.dff, self.DEC_depth, self.DEC_attn_dim, self.DEC_attn_heads,
                                   self.DEC_dropout_rate)

        self.reduc_pos_enc_dec = Sequential([KL.Dense(1),                               #KL.Dense(1, activation='relu'),
                                             KL.Reshape((self.latent_len,), input_shape=(self.latent_len, 1))])

        self.outputadapter = KL.Dense(self.seq_len)

    def call(self, x):

        x = rearrange(x, "a b -> a b 1")
        x = self.embedding_dec(x)
        x += self.positional_enc_dec

        decoded = self.decoder(x)
        decoded = self.reduc_pos_enc_dec(decoded)
        output = self.outputadapter(decoded)

        return output




class FullTransformer(tf.keras.Model):

    def __init__(self,
                 embedding_dim,
                 dff,
                 seq_len,
                 latent_len,
                 enc_depth,
                 enc_attn_dim,
                 enc_attn_heads,
                 enc_dropout_rate,
                 dec_depth,
                 dec_attn_dim,
                 dec_attn_heads,
                 dec_dropout_rate):
        super(FullTransformer, self).__init__()

        self.Embedding_dim = embedding_dim
        self.dff = dff
        self.seq_len = seq_len
        self.latent_len = latent_len

        self.ENC_depth = enc_depth
        self.ENC_attn_dim = enc_attn_dim
        self.ENC_attn_heads = enc_attn_heads
        self.ENC_dropout_rate = enc_dropout_rate

        self.DEC_depth = dec_depth
        self.DEC_attn_dim = dec_attn_dim
        self.DEC_attn_heads = dec_attn_heads
        self.DEC_dropout_rate = dec_dropout_rate


        self.Encoder = Encoder(embedding_dim=self.Embedding_dim,
                               dff=self.dff,
                               seq_len=self.seq_len,
                               latent_len=self.latent_len,
                               enc_depth=self.ENC_depth,
                               enc_attn_dim=self.ENC_attn_dim,
                               enc_attn_heads=self.ENC_attn_heads,
                               enc_dropout_rate=self.ENC_dropout_rate)

        self.Decoder = Decoder(embedding_dim=self.Embedding_dim,
                               dff=self.dff,
                               seq_len=self.seq_len,
                               latent_len=self.latent_len,
                               dec_depth=self.DEC_depth,
                               dec_attn_dim=self.DEC_attn_dim,
                               dec_attn_heads=self.DEC_attn_heads,
                               dec_dropout_rate=self.DEC_dropout_rate)


    def call(self, inputs):

        latent = self.Encoder(inputs)

        output = self.Decoder(latent)

        return latent, output
