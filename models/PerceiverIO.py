# -*- coding: utf-8 -*-
"""
Definition of the PerceiverIO Class for Spike Sorting
"""
import numpy as np
import math
import tensorflow as tf


from einops import rearrange, repeat
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as KL




def get_angles(pos, i, d_model):
    #args: - pos: is the position of the input embedding in the sequence (hello world, how are you: pos(world)=2)

    # - i: is the dimension of the input in the input embedding ("world" --> [0.1,0.4,0.2,0.3]: i=2 refers to input 0.4 of "world" embedding)

    # - d_model: dimension of the embedding ("world" --> [0.1,0.4,0.2,0.3]: d_model = 4)

    # --> in our setting: no embedding necessary, so always i=1 and d_model=1

    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

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

        #self.fin_norm = KL.LayerNormalization(axis=-1)

        self.attn_mlp = Sequential([KL.LayerNormalization(axis=-1),

                                    KL.Dense(self.dff, activation=tf.keras.activations.gelu),

                                    KL.Dense(self.state_channels),

                                    KL.Dropout(self.dropout_rate)])

    def call(self, inputs):

        #x = inputs

        normed_inputs = self.attn_norm1(inputs)

        x = self.attn(normed_inputs, normed_inputs, return_attention_scores=False)

        x += self.attn_mlp(x)

        #x = self.fin_norm(x)

        return x


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

        #self.fin_norm = KL.LayerNormalization(axis=-1)

        self.x_attn_mlp = Sequential([KL.LayerNormalization(axis=-1),

                                      KL.Dense(self.dff, activation=tf.keras.activations.gelu),

                                      KL.Dense(self.state_channels),

                                      KL.Dropout(self.dropout_rate)])

    def call(self, inputs):
        q, k = inputs

        q_norm = self.x_attn_norm1(q)

        k_norm = self.x_attn_norm2(k)

        x = self.x_attn(q_norm, k_norm, return_attention_scores=False)  # q + self.x_attn(q_norm, k_norm, return_attention_scores=False)

        x += self.x_attn_mlp(x)

        #x = self.fin_norm(x)

        return x


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
    
        super(AttentionBlock,self).__init__()

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

        #self.state_stack = tf.TensorArray(tf.float32,

                                             # size=self.depth,

                                             # dynamic_size=False,

                                             # clear_after_read=False)

        self.iter = tf.constant(1)

    

    def call(self, latents, inputs):

        latents = self.CrossAttention([latents, inputs])

        i = self.iter

        #state_stack = self.state_stack

        state_0 = self.SelfAttention(latents)

        #state_stack = state_stack.write(0, state_0)

        #cond = lambda i, ss: tf.less(i, self.depth)

        #def body(i, state_stack):

            #state = self.SelfAttention(state_stack.read(i-1))

            #state_stack = state_stack.write(i, state)

            #return (i+1, state_stack)

        #[i, state_stack] = tf.while_loop(cond,  body, [i, state_stack])

        while i < self.depth:

            state_0 = self.SelfAttention(state_0)

            i += 1

        #state_out = state_stack.read(i-1)

        return state_0


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


class Encoder(tf.keras.Model):

    def __init__(self,

                 embedding_dim=65,

                 seq_len = 64,

                 latent_len = 256,

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
    
        super(Encoder,self).__init__()

        self.Embedding_dim = embedding_dim

        self.seq_len = seq_len

        self.latent_len =latent_len

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


        self.project_inp = Sequential([KL.Dense(self.Embedding_dim)])

        self.positional_enc_inp = positional_encoding(self.seq_len, self.Embedding_dim)

        self.positional_enc_state = positional_encoding(self.state_index, self.state_channels)

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

                                KL.LayerNormalization(axis=-1, trainable=False)])

        self.logits_to_latent = KL.Dense(self.latent_len)

    def call(self, inputs):

        inputs = rearrange(inputs, "a b -> a b 1")

        inputs = self.project_inp(inputs)

        inputs += self.positional_enc_inp

        ENC_state = self.ENC_state

        ENC_state += self.positional_enc_state

        ENC_state = self.AttentionPipe(ENC_state, inputs)

        logits = self.logs(ENC_state)

        logits = self.logits_to_latent(logits)

        return logits


class Decoder(tf.keras.Model):

    def __init__(self,

                 embedding_dim=65,

                 seq_len=64,

                 latent_len=256,

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
    
        super(Decoder, self).__init__()

        self.Embedding_dim = embedding_dim

        self.seq_len = seq_len

        self.latent_len = latent_len

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

        self.project_inp = Sequential([KL.Dense(self.Embedding_dim)])

        self.positional_inp = positional_encoding(self.latent_len, self.Embedding_dim)

        self.positional_state = positional_encoding(self.state_index, self.state_channels)

        self.state = tf.clip_by_value(tf.Variable(tf.random.normal((self.state_index,

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

        self.reshaping = KL.Reshape((self.state_index*self.state_channels,), input_shape=(self.state_index, self.state_channels))

        self.outputadapter = KL.Dense(self.seq_len)

    def call(self, inputs):

        inputs = rearrange(inputs, "a b -> a b 1")

        inputs = self.project_inp(inputs)

        inputs += self.positional_inp

        state = self.state

        state += self.positional_state

        state = self.AttentionPipe(state, inputs)

        state = self.reshaping(state)

        out = self.outputadapter(state)

        return out



class AutoPerceiver(tf.keras.Model):

    def __init__(self,

                 embedding_dim=65,

                 seq_len = 64,

                 latent_len = 256,

                 ENC_number_of_layers=3,

                 ENC_state_index=256,

                 ENC_state_channels=128,

                 ENC_dff=128,

                 ENC_x_attn_dim=64,

                 ENC_x_attn_heads=8,

                 ENC_depth=4,

                 ENC_attn_dim=64,

                 ENC_attn_heads=8,

                 ENC_dropout_rate=0.1,

                 DEC_number_of_layers=3,

                 DEC_state_index=256,

                 DEC_state_channels=128,

                 DEC_dff=128,

                 DEC_x_attn_dim=64,

                 DEC_x_attn_heads=8,

                 DEC_depth=4,

                 DEC_attn_dim=64,

                 DEC_attn_heads=8,

                 DEC_dropout_rate=0.1):

        super(AutoPerceiver,self).__init__()

        self.embedding_dim = embedding_dim

        self.seq_len = seq_len

        self.latent_len = latent_len

        self.ENC_number_of_layers = ENC_number_of_layers

        self.ENC_state_index = ENC_state_index

        self.ENC_state_channels = ENC_state_channels

        self.ENC_dff = ENC_dff

        self.ENC_x_attn_dim = ENC_x_attn_dim

        self.ENC_x_attn_heads = ENC_x_attn_heads

        self.ENC_depth = ENC_depth

        self.ENC_attn_dim = ENC_attn_dim

        self.ENC_attn_heads = ENC_attn_heads

        self.ENC_dropout_rate = ENC_dropout_rate

        self.DEC_number_of_layers = DEC_number_of_layers

        self.DEC_state_index = DEC_state_index

        self.DEC_state_channels = DEC_state_channels

        self.DEC_dff = DEC_dff

        self.DEC_x_attn_dim = DEC_x_attn_dim

        self.DEC_x_attn_heads = DEC_x_attn_heads

        self.DEC_depth = DEC_depth

        self.DEC_attn_dim = DEC_attn_dim

        self.DEC_attn_heads = DEC_attn_heads

        self.DEC_dropout_rate = DEC_dropout_rate


        self.Encoder = Encoder(embedding_dim=self.embedding_dim,

                               seq_len=self.seq_len,

                               latent_len=self.latent_len,

                               number_of_layers=self.ENC_number_of_layers,

                               state_index=self.ENC_state_index,

                               state_channels=self.ENC_state_channels,

                               dff=self.ENC_dff,

                               x_attn_dim=self.ENC_x_attn_dim,

                               x_attn_heads=self.ENC_x_attn_heads,

                               depth=self.ENC_depth,

                               attn_dim=self.ENC_attn_dim,

                               attn_heads=self.ENC_attn_heads,

                               dropout_rate=self.ENC_dropout_rate)

        self.Decoder = Decoder(embedding_dim=self.embedding_dim,

                               seq_len=self.seq_len,

                               latent_len=self.latent_len,

                               number_of_layers=self.DEC_number_of_layers,

                               state_index=self.DEC_state_index,

                               state_channels=self.DEC_state_channels,

                               dff=self.DEC_dff,

                               x_attn_dim=self.DEC_x_attn_dim,

                               x_attn_heads=self.DEC_x_attn_heads,

                               depth=self.DEC_depth,

                               attn_dim=self.DEC_attn_dim,

                               attn_heads=self.DEC_attn_heads,

                               dropout_rate=self.DEC_dropout_rate)

    def call(self, inputs):

        logits = self.Encoder(inputs)

        output = self.Decoder(logits)

        return logits, output
