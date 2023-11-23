from pathlib import Path
import tensorflow as tf
from custom_layers import DecoderLayer, SeqEmbedding, ImageEmbedding, EncoderLayer
from dataset import detokenise
from PIL import Image


class OCRModel(tf.keras.Model):
    
    def __init__(self, output_layer,
                 dictionary, 
                 embedding_weights,
                 image_shape=[100, 2000], # HxW
                 patch_shape=[25, 25],
                 vocab_size=409094, 
                 num_heads=4,
                 num_layers=2, 
                 units=256,
                 patches_length=2000, 
                 max_length=64, 
                 dropout_rate=0.1
                 ):
        
        super().__init__()

        self.max_length = max_length
        self.image_shape = image_shape
        self.patch_shape = patch_shape

        self.seq_embedding = SeqEmbedding(embedding_weights, vocab_size, max_length, units)

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
        img = Image.open(image).convert('RGB')
        initial = self.word_to_token(['<s>']) # (batch, sequence)
        timg = tf.keras.utils.img_to_array(img)

        #TODO move all this stuff into a preprocessing function
        timg = tf.image.resize_with_pad(timg, target_height=self.image_shape[0], target_width=self.image_shape[1])
        timg = tf.image.rgb_to_grayscale(timg)
        timg = tf.cast(timg, tf.float32) * tf.constant(1/255.) 
        timg = timg[tf.newaxis, :]
        patches = tf.image.extract_patches(timg, 
                                    sizes=[1, self.patch_shape[0], self.patch_shape[1], 1], 
                                    strides=[1, self.patch_shape[0], self.patch_shape[1], 1], 
                                    rates=[1, 1, 1, 1], 
                                    padding="VALID")

        patches = tf.reshape(patches, (patches.shape[0], patches.shape[1] * patches.shape[2], patches.shape[-1]))
        tokens = initial # (batch, sequence)
        for n in range(self.max_length // 2):
            paddings = tf.constant([[0, self.max_length - len(tokens)]])
            preds = self((patches, tf.cast(tf.pad(tokens, paddings)[tf.newaxis, :], dtype=tf.int64))).numpy()  # (batch, sequence, vocab)
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


    