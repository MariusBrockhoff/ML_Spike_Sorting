# -*- coding: utf-8 -*-

import numpy as np
import math



from einops import rearrange, repeat
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as KL


class FourierEncoding(tf.keras.layers.Layer):
    # input shape: (batch_size, seq_length, n_features, 1)
    # output shape: (batch_size, seq_length*n_features, d_model)

    def __init__(self, max_freq, num_bands, base):
        super(FourierEncoding, self).__init__()

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

            dtype=tf.float32)

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


def get_angles(pos, i, d_model):
    # args: - pos: is the position of the input embedding in the sequence (hello world, how are you: pos(world)=2)

    # - i: is the dimension of the input in the input embedding ("world" --> [0.1,0.4,0.2,0.3]: i=2 refers to input 0.4 of "world" embedding)

    # - d_model: dimension of the embedding ("world" --> [0.1,0.4,0.2,0.3]: d_model = 4)

    # --> in our setting: no embedding necessary, so always i=1 and d_model=1

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    x = tf.cast(pos_encoding, dtype=tf.float32)

    x = tf.squeeze(x)

    return x


class SelfAttention(tf.keras.Model):

    def __init__(self,

                 state_index=64,

                 state_channels=128,

                 dff=128,

                 attn_dim=64,

                 attn_heads=8,

                 dropout_rate=0.1):
        super(SelfAttention, self).__init__()

        self.state_index = state_index

        self.state_channels = state_channels

        self.dff = dff

        self.attn_dim = attn_dim

        self.attn_heads = attn_heads

        self.dropout_rate = dropout_rate

        self.attn = KL.MultiHeadAttention(self.attn_heads, self.attn_dim)

        self.attn_sum1 = KL.Add()

        self.attn_norm1 = KL.LayerNormalization(axis=-1)

        self.attn_mlp = Sequential([KL.LayerNormalization(axis=-1),

                                    KL.Dense(self.dff, activation=tf.keras.activations.gelu),
                                    # , activity_regularizer=RegL1(0.01)),

                                    KL.Dropout(self.dropout_rate),

                                    KL.Dense(self.state_channels)])  # , activity_regularizer=RegL1(0.01))])

    def call(self, inputs):
        normed_inputs = self.attn_norm1(inputs)

        state = self.attn(normed_inputs, normed_inputs, return_attention_scores=False)

        state = self.attn_sum1([inputs, state])

        state_mlp = self.attn_mlp(state)

        return state_mlp


class CrossAttention(tf.keras.Model):

    def __init__(self,

                 state_index=256,

                 state_channels=128,

                 dff=128,

                 x_attn_dim=64,

                 x_attn_heads=8,

                 dropout_rate=0.1):
        super(CrossAttention, self).__init__()

        self.state_index = state_index

        self.state_channels = state_channels

        self.dff = dff

        self.x_attn_dim = x_attn_dim

        self.x_attn_heads = x_attn_heads

        self.dropout_rate = dropout_rate

        self.x_attn = KL.MultiHeadAttention(self.x_attn_heads, self.x_attn_dim)

        self.x_attn_norm1 = KL.LayerNormalization(axis=-1)

        self.x_attn_norm2 = KL.LayerNormalization(axis=-1)

        self.x_attn_mlp = Sequential([KL.LayerNormalization(axis=-1),

                                      KL.Dense(self.dff, activation=tf.keras.activations.gelu),
                                      # , activity_regularizer=RegL1(0.01)),

                                      KL.Dropout(self.dropout_rate),

                                      KL.Dense(self.state_channels)])  # , activity_regularizer=RegL1(0.01))])

    def call(self, inputs):
        q, k = inputs

        q_norm = self.x_attn_norm1(q)

        k_norm = self.x_attn_norm2(k)

        state = self.x_attn(q_norm, k_norm, return_attention_scores=False)

        state_mlp = self.x_attn_mlp(state)

        return state_mlp

class AttentionBlock(tf.keras.layers.Layer):

    def __init__(self,

                 state_index=256,

                 state_channels=128,

                 dff=128,

                 x_attn_dim=64,

                 x_attn_heads=8,

                 depth=4,

                 attn_dim=64,

                 attn_heads=8,

                 dropout_rate=0.1):
        super(AttentionBlock, self).__init__()

        self.state_index = state_index

        self.state_channels = state_channels

        self.dff = dff

        self.x_attn_dim = x_attn_dim

        self.x_attn_heads = x_attn_heads

        self.depth = depth

        self.attn_dim = attn_dim

        self.attn_heads = attn_heads

        self.dropout_rate = dropout_rate

        self.CrossAttention = CrossAttention(state_index=self.state_index,

                                             state_channels=self.state_channels,

                                             dff=self.dff,

                                             x_attn_dim=self.x_attn_dim,

                                             x_attn_heads=self.x_attn_heads,

                                             dropout_rate=self.dropout_rate)

        self.SelfAttention = SelfAttention(state_index=self.state_index,

                                           state_channels=self.state_channels,

                                           dff=self.dff,

                                           attn_dim=self.attn_dim,

                                           attn_heads=self.attn_heads,

                                           dropout_rate=self.dropout_rate)

        self.state_stack = tf.TensorArray(tf.float32,

                                          size=self.depth,

                                          dynamic_size=False,

                                          clear_after_read=False)


        self.iter = tf.constant(1)

    def call(self, latents, inputs):

        latents = self.CrossAttention([latents, inputs])

        i = self.iter

        while i <= self.depth:

            latents = self.SelfAttention(latents)

            i += 1

        state_out = latents

        return state_out


class AttentionPipe(tf.keras.layers.Layer):

    def __init__(self,

                 number_of_layers=3,

                 state_index=256,

                 state_channels=128,

                 dff=128,

                 x_attn_dim=64,

                 x_attn_heads=8,

                 depth=4,

                 attn_dim=64,

                 attn_heads=8,

                 dropout_rate=0.1):
        super(AttentionPipe, self).__init__()

        self.number_of_layers = number_of_layers

        self.state_index = state_index

        self.state_channels = state_channels

        self.dff = dff

        self.x_attn_dim = x_attn_dim

        self.x_attn_heads = x_attn_heads

        self.depth = depth

        self.attn_dim = attn_dim

        self.attn_heads = attn_heads

        self.dropout_rate = dropout_rate

        self.Attention_blocks = [AttentionBlock(state_index=self.state_index,

                                                state_channels=self.state_channels,

                                                dff=self.dff,

                                                x_attn_dim=self.x_attn_dim,

                                                x_attn_heads=self.x_attn_heads,

                                                depth=self.depth,

                                                attn_dim=self.attn_dim,

                                                attn_heads=self.attn_heads,

                                                dropout_rate=self.dropout_rate) for _ in range(self.number_of_layers)]

    def call(self, latents, inputs):
        b, *_ = inputs.shape

        x = repeat(latents, 'i c -> b i c', b=b)

        for i in range(self.number_of_layers):

            x = self.Attention_blocks[i](x, inputs)

        return x


class Perceiver(tf.keras.Model):

    def __init__(self,

                 Embedding_dim=65,

                 seq_len=64,

                 number_of_layers=3,

                 latent_len=30,

                 state_index=256,

                 state_channels=128,

                 dff=128,

                 x_attn_dim=64,

                 x_attn_heads=8,

                 depth=4,

                 attn_dim=64,

                 attn_heads=8,

                 dropout_rate=0.1):
        super(Perceiver, self).__init__()

        self.Embedding_dim = Embedding_dim

        self.seq_len = seq_len

        self.number_of_layers = number_of_layers

        self.latent_len = latent_len

        self.state_index = state_index

        self.state_channels = state_channels

        self.dff = dff

        self.x_attn_dim = x_attn_dim

        self.x_attn_heads = x_attn_heads

        self.depth = depth

        self.attn_dim = attn_dim

        self.attn_heads = attn_heads

        self.dropout_rate = dropout_rate

        self.project_inp = Sequential([KL.Dense(self.Embedding_dim)])

        self.positional_enc_inp = positional_encoding(self.seq_len, self.Embedding_dim)

        self.ENC_state = tf.clip_by_value(tf.Variable(tf.random.normal((self.state_index,

                                                                        self.state_channels), mean=0.0, stddev=0.02),

                                                      trainable=True,

                                                      dtype=tf.float32), -2.0, 2.0)

        self.AttentionPipe = AttentionPipe(number_of_layers=self.number_of_layers,

                                           state_index=self.state_index,

                                           state_channels=self.state_channels,

                                           dff=self.dff,

                                           x_attn_dim=self.x_attn_dim,

                                           x_attn_heads=self.x_attn_heads,

                                           depth=self.depth,

                                           attn_dim=self.attn_dim,

                                           attn_heads=self.attn_heads,

                                           dropout_rate=self.dropout_rate)

        self.logs = Sequential([KL.Lambda(lambda x: tf.reduce_mean(x, axis=-2), trainable=False),

                                KL.Dense(self.latent_len)])


    def call(self, inputs):
        inputs = rearrange(inputs, "a b -> a b 1")

        inputs = self.project_inp(inputs)

        inputs += self.positional_enc_inp

        ENC_state = self.ENC_state

        ENC_state = self.AttentionPipe(ENC_state, inputs)

        logits = self.logs(ENC_state)

        return ENC_state, logits
