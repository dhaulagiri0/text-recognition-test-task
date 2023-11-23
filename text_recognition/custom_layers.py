import tensorflow as tf

import numpy as np
import tqdm, collections

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

# class PositionalEmbedding(tf.keras.layers.Layer):
#   def __init__(self, vocab_size, d_model):
#     super().__init__()
#     self.d_model = d_model
#     self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
#     self.pos_encoding = positional_encoding(length=2048, depth=d_model)

#   def compute_mask(self, *args, **kwargs):
#     return self.embedding.compute_mask(*args, **kwargs)

#   def call(self, x):
#     length = tf.shape(x)[1]
#     x = self.embedding(x)
#     # This factor sets the relative scale of the embedding and positonal_encoding.
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#     x = x + self.pos_encoding[tf.newaxis, :length, :]
#     return x

class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_weights, vocab_size=409094, max_length=64, depth=256):
        super().__init__()
        self.pos_encoding = positional_encoding(length=2048, depth=depth)

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=depth,
            mask_zero=True,)
        
        #embeddings_initializer=tf.keras.initializers.Constant(embedding_weights),
        
        self.depth = depth
        self.add = tf.keras.layers.Add()

    def compute_mask(self, *args, **kwargs):
        return self.token_embedding.compute_mask(*args, **kwargs)

    def call(self, seq):
        length = tf.shape(seq)[1]
        x = self.token_embedding(seq) # (batch, seq, depth)

        # x = tf.range(tf.shape(seq)[1])  # (seq)
        # x = x[tf.newaxis, :]  # (1, seq)
        # x = self.pos_embedding(x)  # (1, seq, depth)

        x *= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        # return self.add([seq,x])
        return x
    
class ImageEmbedding(tf.keras.layers.Layer):
    def __init__(self, patches_length=320, units=256):
        super().__init__()
        self.patches_length = patches_length
        self.units = units
        self.pos_encoding = positional_encoding(length=2048, depth=units)

        #Nx80x100 -> Nx80x256
        self.image_embedding = tf.keras.layers.Dense(units=units, activation="relu")

        self.add = tf.keras.layers.Add()

    def call(self, seq):
        length = tf.shape(seq)[1]
        x = self.image_embedding(seq) # (batch, seq, depth)

        # x = tf.range(tf.shape(seq)[1])  # (seq)
        # x = x[tf.newaxis, :]  # (1, seq)
        # x = self.pos_embedding(x)  # (1, seq, depth)
        # x = tf.broadcast_to(x, [seq.shape[0], self.patches_length, self.units])
        x *= tf.math.sqrt(tf.cast(self.units, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        # return self.add([seq, x])
        return x
  
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

        out_seq = self.self_attention(in_seq)

        out_seq = self.ff(out_seq)

        return out_seq

class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, vocab_size, vocab_dict, banned_tokens=('<unk>', '<s>'), **kwargs):
        super().__init__()

        self.dense = tf.keras.layers.Dense(
            units=vocab_size, **kwargs)
        self.banned_tokens = banned_tokens
        self.vocab_size = vocab_size
        self.bias = None
        self.vocab_dict = vocab_dict

    def adapt(self, ds):
        counts = collections.Counter()
        vocab_dict = self.vocab_dict

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

        return x + self.bias