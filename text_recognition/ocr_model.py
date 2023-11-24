from pathlib import Path
import tensorflow as tf
from custom_layers import DecoderLayer, SeqEmbedding, ImageEmbedding, EncoderLayer
from dataset import detokenise
from tf_data_process import add_padding
from PIL import Image


class OCRModel(tf.keras.Model):
    
    def __init__(self, output_layer,
                 dictionary, 
                 embedding_weights,
                 image_shape=[100, 2000], # HxW
                 patch_shape=[10, 10],
                 vocab_size=409094, 
                 num_heads=2,
                 num_layers=1, 
                 units=128,
                 max_length=64, 
                 dropout_rate=0.1
                 ):
        
        super().__init__()

        self.max_length = max_length
        self.image_shape = image_shape
        self.patch_shape = patch_shape

        self.seq_embedding = SeqEmbedding(embedding_weights, vocab_size, max_length, units)

        self.image_embedding = ImageEmbedding(image_shape, patch_shape, units)

        self.decoder_layers = [
            DecoderLayer(units, num_heads, dropout_rate) 
            for n in range(num_layers)]

        self.encoder_layers = [
            EncoderLayer(units, num_heads, dropout_rate) 
            for n in range(num_layers)]
        
        self.output_layer = output_layer
        self.word_index = dictionary
        self.index_word = {v:k for k, v in self.word_index.items()}

    def word_to_token(self, pieces):
        tokens = []
        for piece in pieces:
            if piece in self.word_index:
                tokens.append(self.word_index[piece])
            else:
                tokens.append(self.word_index["<unk>"])

        return tokens

    def recognize_text(self, image: Path, temperature=1) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image: The path to the image file.
        :return: The recognized text from the image.
        """
        #TODO move all this stuff into a preprocessing function
        img = Image.open(image).convert('RGB')
        initial = self.word_to_token(['<s>']) # (batch, sequence)
        timg = tf.keras.utils.img_to_array(img)
        timg = tf.cast(timg, tf.float32)
        timg = tf.math.scalar_mul(1/255., timg)[tf.newaxis, :]

        tokens = initial # (batch, sequence)
        for n in range(self.max_length // 2):
            paddings = tf.constant([[0, self.max_length - len(tokens)]])
            preds = self((timg, tf.cast(tf.pad(tokens, paddings)[tf.newaxis, :], dtype=tf.int64))).numpy()  # (batch, sequence, vocab)
            preds = preds[:,-1, :]  #(batch, vocab)
            if temperature==0:
                next = tf.squeeze(tf.argmax(preds, axis=-1))
            else:
                next = tf.squeeze(tf.random.categorical(preds/temperature, num_samples=1))
            tokens = tokens + [next.numpy().tolist()]# (batch, sequence) 

            if next == self.word_index['</s>']:
                break
        
        print(tokens)
        return detokenise(tokens, self.index_word)


    def call(self, inputs):
        patches, tokens = inputs
        image_seq = self.image_embedding(patches)

        txt_seq = self.seq_embedding(tokens)

        # encodes image patches
        for enc_layer in self.encoder_layers:
            image_seq = enc_layer(image_seq)


        # generates text while looking at the image
        for dec_layer in self.decoder_layers:
            txt_seq = dec_layer(inputs=(image_seq, txt_seq))

        txt_seq = self.output_layer(txt_seq)

        return txt_seq


    
