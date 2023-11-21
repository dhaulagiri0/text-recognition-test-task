from pathlib import Path
import tensorflow as tf
from text_recognition.custom_layers import DecoderLayer, SeqEmbedding, ImageEmbedding, EncoderLayer
from text_recognition.embedding_utils import load_json_data
from text_recognition.dataset import detokenise
from text_recognition.tf_data_process import getImageGrey
from PIL import Image


class OCRModel(tf.keras.model):
    
    def __init__(self, output_layer,
                 dictionary, 
                 image_shape=[20, 400], # HxW
                 patch_shape=[10, 10],
                 vocab_size=409094, 
                 num_heads=4,
                 num_layers=2, 
                 units=256,
                 patches_length=80, 
                 max_length=64, 
                 dropout_rate=0.1, 
                 embedding_weigths="dataset/embedding.txt",
                 ):
        
        super().__init__()

        self.max_length = max_length
        self.image_shape = image_shape
        self.patch_shape = patch_shape

        self.seq_embedding = SeqEmbedding(vocab_size, max_length, units, embedding_weigths)

        self.image_embedding = ImageEmbedding(patches_length, units)

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
        for piece in pieces[0]:
            if piece in self.word_index:
                tokens.append(self.word_index[piece])
            else:
                tokens.append(self.word_index["<unk>"])

        tokens.append(self.word_index["</s>"])
        return [tokens]

    def recognize_text(self, image: Path, temperature=1) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image: The path to the image file.
        :return: The recognized text from the image.
        """
        img = Image.open(image)
        initial = self.word_to_index([['[START]']]) # (batch, sequence)

        timg = getImageGrey(img, self.image_shape[0], self.image_shape[1])
        patches = tf.image.extract_patches(timg, 
                                    sizes=[1, self.patch_shape[0], self.patch_shape[1], 1], 
                                    strides=[1, self.patch_shape[0], self.patch_shape[1], 1], 
                                    rates=[1, 1, 1, 1], 
                                    padding="VALID")

        tokens = initial # (batch, sequence)
        for n in range(self.max_length):
            preds = self((patches, tokens)).numpy()  # (batch, sequence, vocab)
            preds = preds[:,-1, :]  #(batch, vocab)
            if temperature==0:
                next = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
            else:
                next = tf.random.categorical(preds/temperature, num_samples=1)  # (batch, 1)
            tokens = tf.concat([tokens, next], axis=1) # (batch, sequence) 

            if next[0] == self.word_to_index('</s>'):
                break

        return detokenise(tokens[0, :], self.index_word)


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


    