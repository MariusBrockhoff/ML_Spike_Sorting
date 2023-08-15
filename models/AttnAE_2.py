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


class EBlock(tf.keras.Model):

    def __init__(self,

                 d_model=65,

                 dff=128,

                 attn_dim=64,

                 attn_heads=8,

                 dropout_rate=0.2):
        super(EBlock, self).__init__()

        self.d_model = d_model

        self.dff = dff

        self.attn_dim = attn_dim

        self.attn_heads = attn_heads

        self.dropout_rate = dropout_rate

        self.attn = KL.MultiHeadAttention(self.attn_heads, self.attn_dim)

        self.attn_sum1 = KL.Add()

        self.attn_norm1 = KL.LayerNormalization(axis=-1)

        self.attn_norm2 = KL.LayerNormalization(axis=-1)

        self.attn_mlp = Sequential([KL.LayerNormalization(axis=-1),

                                    KL.Dense(self.dff, activation=tf.keras.activations.gelu),

                                    KL.Dense(self.d_model)])

    def call(self, inputs):
        # inputs = self.attn_norm1(inputs)

        state, vis = self.attn(inputs, inputs, return_attention_scores=True)

        state = self.attn_norm1(self.attn_sum1([inputs, state]))

        state_mlp = self.attn_mlp(state)

        state_mlp = self.attn_norm2(self.attn_sum1([state, state_mlp]))

        # layernorm

        return state_mlp


class AttnEncoder(tf.keras.layers.Layer):

    def __init__(self,

                 d_model=65,

                 dff=128,

                 ENC_depth=8,

                 attn_dim=64,

                 attn_heads=8,

                 dropout_rate=0.2):
        super(AttnEncoder, self).__init__()

        self.d_model = d_model

        self.dff = dff

        self.ENC_depth = ENC_depth

        self.ENC_attn_dim = attn_dim

        self.ENC_attn_heads = attn_heads

        self.ENC_dropout_rate = dropout_rate

        self.enc_layers = [EBlock(d_model=self.d_model,

                                  dff=self.dff,

                                  attn_dim=self.ENC_attn_dim,

                                  attn_heads=self.ENC_attn_heads,

                                  dropout_rate=self.ENC_dropout_rate) for _ in range(self.ENC_depth)]

    def call(self, x):
        for i in range(self.ENC_depth):
            x = self.enc_layers[i](x)
            #print('shape after encoder layer:', x.shape)

        return x


class Encoder(tf.keras.Model):
    def __init__(self,
                 d_model,
                 dff,
                 seq_len,
                 latent_len,
                 ENC_depth,
                 ENC_attn_dim,
                 ENC_attn_heads,
                 ENC_dropout_rate,
                 reg_value):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.dff = dff
        self.seq_len = seq_len
        self.latent_len = latent_len
        self.ENC_depth = ENC_depth
        self.ENC_attn_dim = ENC_attn_dim
        self.ENC_attn_heads = ENC_attn_heads
        self.ENC_dropout_rate = ENC_dropout_rate
        self.reg_value = reg_value

        self.embedding = Sequential([KL.Dense(self.d_model)])  # Add normalization (*sqrt(embedding_dim))
        self.positional_enc = positional_encoding(self.seq_len, self.d_model)
        self.encoder = AttnEncoder(self.d_model, self.dff, self.ENC_depth, self.ENC_attn_dim, self.ENC_attn_heads,
                               self.ENC_dropout_rate)

        self.reduc_pos_enc = Sequential(
            [KL.Dense(1, activation='relu'),

             KL.Reshape((self.seq_len,), input_shape=(self.seq_len, 1))])  # PREVIOUS: self.reduc_pos_enc = Sequential(
        # [KL.Dense(1, activation='relu', activity_regularizer=tf.keras.regularizers.l1(self.reg_value)),
        # KL.Reshape((self.seq_len,), input_shape=(self.seq_len, 1))])

        self.ENC_to_logits = Sequential([KL.Dense(self.latent_len,
                                                  activation='relu')])  # PREVIOUS: self.ENC_to_logits = Sequential([KL.Dense(self.latent_len, activation='relu',
        # activity_regularizer=tf.keras.regularizers.l1(self.reg_value))])

    def call(self, inputs):

        inputs = rearrange(inputs, "a b -> a b 1")
        inputs = self.embedding(inputs)
        # inputs = self.fourier_enc(inputs)
        inputs += self.positional_enc

        # Encoder
        encoded = self.encoder(inputs)
        encoded = self.reduc_pos_enc(encoded)
        latents = self.ENC_to_logits(encoded)

        return latents


class Decoder(tf.keras.Model):
    def __init__(self,
                 DEC_layers,
                 seq_len):
        super(Decoder, self).__init__()

        self.DEC_layers = DEC_layers
        self.seq_len = seq_len

        self.decoder_layers = [KL.Dense(self.DEC_layers[i], kernel_initializer='glorot_uniform') for i in
                               range(len(self.DEC_layers))]

        self.outputadapter = Sequential([KL.Dense(self.seq_len)])

    def call(self, inputs):

        x = inputs

        # Decoder
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x)

        output = self.outputadapter(x)

        return output





class Attention_AE(tf.keras.Model):

    def __init__(self,
                 d_model,
                 dff,
                 seq_len,
                 latent_len,
                 ENC_depth,
                 ENC_attn_dim,
                 ENC_attn_heads,
                 ENC_dropout_rate,
                 DEC_layers,
                 reg_value):
        super(Attention_AE, self).__init__()

        self.d_model = d_model
        self.dff = dff
        self.seq_len = seq_len
        self.latent_len = latent_len
        self.ENC_depth = ENC_depth
        self.ENC_attn_dim = ENC_attn_dim
        self.ENC_attn_heads = ENC_attn_heads
        self.ENC_dropout_rate = ENC_dropout_rate
        self.DEC_layers = DEC_layers
        self.reg_value = reg_value

        self.Encoder = Encoder(d_model=self.d_model,
                               dff=self.dff,
                               seq_len=self.seq_len,
                               latent_len=self.latent_len,
                               ENC_depth=self.ENC_depth,
                               ENC_attn_dim=self.ENC_attn_dim,
                               ENC_attn_heads=self.ENC_attn_heads,
                               ENC_dropout_rate=self.ENC_dropout_rate,
                               reg_value=self.reg_value)

        self.Decoder = Decoder(DEC_layers=self.DEC_layers,
                               seq_len=self.seq_len)

    def call(self, inputs):

        latent_vec = self.Encoder(inputs)

        final_output = self.Decoder(latent_vec)

        return latent_vec, final_output
