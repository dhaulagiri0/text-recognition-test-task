import tensorflow as tf

import numpy as np
import tqdm, collections, einops

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


class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_weights, vocab_size=10000, max_length=32, depth=300):
        super().__init__()

        # self.position_encoding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth)
        self.position_encoding = positional_encoding(length=max_length, depth=depth)

        hits = 0
        embedding_matrix = np.zeros((vocab_size, 300))
        for i ,vector in embedding_weights.items():
            embedding_matrix[i] = vector
            hits += 1
        print(f"Converted {hits} words")

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=300,
            mask_zero=True,
            weights=[embedding_matrix],
            trainable=True)
        #embeddings_initializer=tf.keras.initializers.Constant(embedding_weights),

        self.depth = depth
        self.max_length = max_length
        self.add = tf.keras.layers.Add()

    def compute_mask(self, *args, **kwargs):
        return self.token_embedding.compute_mask(*args, **kwargs)

    def call(self, seq):
        length = tf.shape(seq)[1]
        seq = self.token_embedding(seq) # (batch, seq, depth)
        seq = tf.pad(seq, paddings=[[0, 0], [0, 0], [0, self.depth - tf.shape(seq)[-1]]])

        # x = tf.range(tf.shape(seq)[1])  # (seq)
        # x = x[tf.newaxis, :]  # (1, seq)
        # x = self.position_encoding(x)  # (1, seq, depth)

        seq *= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        x    = self.position_encoding[tf.newaxis, :length, :]

        return self.add([seq, x])
        # return x

class ImageEmbedding(tf.keras.layers.Layer):
    def __init__(self, image_shape=[75, 450], patch_shape=(25, 25), depth=256):
        super().__init__()
        self.depth = depth

        self.projection = tf.keras.layers.Dense(units=depth, activation="linear")
        # self.image_embedding = tf.keras.layers.Conv2D(depth, patch_shape, patch_shape, activation="linear")
        self.num_patches = (image_shape[0] // patch_shape[0]) * (image_shape[1] // patch_shape[1])

        self.position_encoding = tf.keras.layers.Embedding(input_dim=self.num_patches, output_dim=depth)
        # self.position_encoding = positional_encoding(length=self.num_patches, depth=depth)

        self.extractor = tf.keras.applications.MobileNetV3Small(input_shape=(image_shape[0], image_shape[1], 3),
                                               include_top=False,
                                               weights='imagenet')

        self.add = tf.keras.layers.Add()
        self.depth = depth
        self.patch_shape = patch_shape

    def call(self, seq):
        length = tf.shape(seq)[1]
        patch_shape = self.patch_shape
        # seq = self.image_embedding(seq) # (batch, seq, depth)
        # seq = tf.image.extract_patches(
        #                 images =seq,
        #                 sizes  =[1, patch_shape[0], patch_shape[1], 1],
        #                 strides=[1, patch_shape[0], patch_shape[1], 1],
        #                 rates  =[1, 1, 1, 1],
        #                 padding='VALID' )
        
        seq = self.extractor(seq)
        seq = einops.rearrange(seq, 'b h w c -> b (h w c)')
        
        # seq = tf.keras.layers.Reshape((self.num_patches, seq.shape[-1]))(seq)
        seq = self.projection(seq)

        x = tf.range(tf.shape(seq)[1])  # (seq)
        x = x[tf.newaxis, :]  # (1, seq)
        x = self.position_encoding(x)  # (1, seq, depth)

        # seq *= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        # x    = self.position_encoding[tf.newaxis, :length, :]

        return self.add([seq, x])
        # return x
  
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

#for the encoder
class GlobalSelfAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        attn = self.mha(query=x, value=x,
                        key=x, use_causal_mask=False)
            
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
        attn = self.mha(
                value=y, query=x)

        x = self.add([x, attn])
        return self.layernorm(x)
  
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units=2*units, activation='gelu'),
            tf.keras.layers.Dense(units=units),
            tf.keras.layers.Dropout(rate=dropout_rate),
        ])
        self.add = tf.keras.layers.Add()

        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)]) 
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


    def call(self, inputs):
        # in_seq is the image, out_seq is the tokens
        in_seq, out_seq = inputs

        out_seq = self.self_attention(out_seq)

        out_seq = self.cross_attention(out_seq, in_seq)

        out_seq = self.ff(out_seq)

        return out_seq

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(num_heads=num_heads,
                                                key_dim=units,
                                                dropout=dropout_rate)
        
        self.ff = FeedForward(units=units, dropout_rate=0)


    def call(self, inputs):
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
        self.add = tf.keras.layers.Add()
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
