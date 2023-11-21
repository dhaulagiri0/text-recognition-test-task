import tensorflow as tf
from text_recognition import embedding_utils as eu

import numpy as np
import tqdm, collections


class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size=409094, max_length=64, depth=256, embedding_weights="dataset/embedding.txt"):
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth)

        embedding, dim = eu.load_embedding(embedding_weights)
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=dim,
            weight=[embedding],
            mask_zero=True)

        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq = self.token_embedding(seq) # (batch, seq, depth)

        x = tf.range(tf.shape(seq)[1])  # (seq)
        x = x[tf.newaxis, :]  # (1, seq)
        x = self.pos_embedding(x)  # (1, seq, depth)

        return self.add([seq,x])
    
class ImageEmbedding(tf.keras.layers.Layer):
    def __init__(self, patches_length=80, units=256):
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=patches_length, output_dim=units)

        #Nx80x100 -> Nx80x256
        self.image_embedding = tf.keras.layers.Dense(units=256, activation="relu")

        self.add = tf.keras.layers.Add()

    def call(self, seq):
        seq = self.image_embedding(seq) # (batch, seq, depth)

        x = tf.range(tf.shape(seq)[1])  # (seq)
        x = x[tf.newaxis, :]  # (1, seq)
        x = self.pos_embedding(x)  # (1, seq, depth)

        return self.add([seq,x])
  
class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        # Use Add instead of + so the keras mask propagates through.
        self.add = tf.keras.layers.Add() 
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        # query = value = x so this is a form of self attention
        attn = self.mha(query=x, value=x,
                        use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)
    
# looks at both the image and the tokens
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add() 
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, y, **kwargs):
        # x is the tokens, y is the image
        attn, attention_scores = self.mha(
                query=x, value=y,
                return_attention_scores=True)

        self.last_attention_scores = attention_scores

        x = self.add([x, attn])
        return self.layernorm(x)
  
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units=2*units, activation='relu'),
            tf.keras.layers.Dense(units=units),
            tf.keras.layers.Dropout(rate=dropout_rate),
        ])

        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.seq(x)
        return self.layernorm(x)
  
# the decoder block in the original transformer paper
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()

        self.self_attention = CausalSelfAttention(num_heads=num_heads,
                                                key_dim=units,
                                                dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads,
                                            key_dim=units,
                                            dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)


    def call(self, inputs, training=False):
        # in_seq is the image, out_seq is the tokens
        in_seq, out_seq = inputs

        out_seq = self.self_attention(out_seq)

        out_seq = self.cross_attention(out_seq, in_seq)

        self.last_attention_scores = self.cross_attention.last_attention_scores

        out_seq = self.ff(out_seq)

        return out_seq

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()

        self.self_attention = CausalSelfAttention(num_heads=num_heads,
                                                key_dim=units,
                                                dropout=dropout_rate)
        
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)


    def call(self, inputs, training=False):
        # in_seq is the image, out_seq is the tokens
        in_seq = inputs

        out_seq = self.self_attention(out_seq)

        out_seq = self.ff(out_seq)

        return out_seq

class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, vocab_size, banned_tokens=('', '<UNK>', '<s>'), **kwargs):
        super().__init__()

        self.dense = tf.keras.layers.Dense(
            units=vocab_size, **kwargs)
        self.banned_tokens = banned_tokens
        self.vocab_size = vocab_size
        self.bias = None

    def adapt(self, ds):
        counts = collections.Counter()
        vocab_dict = eu.load_json_data("dataset/vocab_bpemb.json")

        for tokens in tqdm.tqdm(ds):
            counts.update(tokens.numpy().flatten())

        counts_arr = np.zeros(shape=(self.vocab_size,))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())

        counts_arr = counts_arr[:]
        for token in self.banned_tokens:
            counts_arr[vocab_dict[token]] = 0

        total = counts_arr.sum()
        p = counts_arr/total
        p[counts_arr==0] = 1.0
        log_p = np.log(p)  # log(1) == 0

        entropy = -(log_p*p).sum()

        print()
        print(f"Uniform entropy: {np.log(self.vocab_size):0.2f}")
        print(f"Marginal entropy: {entropy:0.2f}")

        self.bias = log_p
        self.bias[counts_arr==0] = -1e9

    def call(self, x):
        x = self.dense(x)
        # TODO(b/250038731): Fix this.
        # An Add layer doesn't work because of the different shapes.
        # This clears the mask, that's okay because it prevents keras from rescaling
        # the losses.
        return x + self.bias