from pathlib import Path
import tensorflow as tf
from custom_layers import DecoderLayer, SeqEmbedding, ImageEmbedding, EncoderLayer
from dataset import detokenise
from tf_data_process import TfDataProcessor
from PIL import Image


class OCRModel(tf.keras.Model):
    
    def __init__(self, output_layer,
                 dictionary, 
                 embedding_weights,
                 image_shape=[150, 450], # HxW
                 patch_shape=[10, 10],
                 vocab_size=3725, 
                 num_heads=8,
                 num_layers=4, 
                 units=512,
                 max_length=32, 
                 dropout_rate=0.2):
        
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
        self.word_2_token = dictionary
        self.token_2_word = {v : k for k, v in self.word_2_token.items()}

    # converts an entire sequence to tokens
    def words_to_token(self, pieces):
        tokens = []
        for piece in pieces:
            if piece in self.word_2_token:
                tokens.append(self.word_2_token[piece])
            else:
                tokens.append(self.word_2_token["<unk>"])

        return tokens
    
    # expects image to be a path
    def img_preprocess(self, image):

        img = Image.open(image).convert('RGB')
        # padding
        timg = TfDataProcessor.add_padding(img, target_shape=[150, 450])
        # grayscale
        timg = tf.image.rgb_to_grayscale(timg)
        # norm
        timg = tf.math.scalar_mul(1/255., timg)
        
        return timg

    def recognize_text(self, image: Path, temperature=1) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image: The path to the image file.
        :return: The recognized text from the image.
        """
        timg = self.img_preprocess(image)

        initial = self.words_to_token(['<s>']) # (batch, sequence)
        tokens = initial # (batch, sequence)

        for n in range(self.max_length):
            
            # process tokens for input
            paddings = tf.constant([[0, self.max_length - len(tokens)]])
            cur_tok =  tf.cast(tf.pad(tokens, paddings)[tf.newaxis, :], dtype=tf.int64)

            preds = self((timg, cur_tok)).numpy()  # (batch, sequence, vocab)
            preds = preds[:,-1, :]  #(batch, vocab)
            
            if temperature==0:
                next = tf.squeeze(tf.argmax(preds, axis=-1))
            else:
                next = tf.squeeze(tf.random.categorical(preds/temperature, num_samples=1))

            # update tokens
            tokens = tokens + [next.numpy().tolist()]# (batch, sequence) 

            if next == self.word_2_token['</s>']:
                break

        # debug printing
        print(tokens)

        return detokenise(tokens, self.token_2_word)


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


    
