from pathlib import Path
import tensorflow as tf
from custom_layers import DecoderLayer, SeqEmbedding, ImageEmbedding, EncoderLayer
from dataset import detokenise
from tf_data_process import TfDataProcessor
from PIL import Image
import cv2
import numpy as np


class OCRModel(tf.keras.Model):
    
    def __init__(self, output_layer,
                 dictionary, 
                 embedding_weights,
                 image_shape=[150, 450], # HxW
                 patch_shape=[10, 10],
                 vocab_size=10000, 
                 num_heads=8,
                 num_layers=4, 
                 units=512,
                 max_length=32, 
                 dropout_rate=0.5):
        
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
    @staticmethod
    def img_preprocess(image, image_shape):

        # padding
        timg = TfDataProcessor.add_padding(img, target_shape=image_shape)
        # grayscale
        timg = tf.image.rgb_to_grayscale(timg)
        # norm
        timg = tf.math.scalar_mul(1/255., timg)
        
        return timg

    def detect_one(self, image, temperature):

        timg = self.img_preprocess(image, self.image_shape)

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

    def recognize_text(self, image: Path, temperature=1, get_patches=False) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image: The path to the image file.
        :return: The recognized text from the image.
        """
        img = Image.open(image).convert('RGB')
        # divide a large image into text blobs if necessary
        if get_patches:
            image_list = OCRModel.getSubImages(img)
        else:
            image_list = [img]
        
        overall = ""
        for image in image_list:

            tokens = self.detect_one(img, temperature)
            text = detokenise(tokens, self.token_2_word)
            overall = f"{text.strip()}\n" + overall

        return overall.strip("\n")


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

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del txt_seq._keras_mask
        except AttributeError:
            pass
        return txt_seq
    
    # uses some cv2 contouring techniques to get rows of text in a image from the bottom up
    def getSubImages(img, padding=0.0):

        (H, W) = img.shape[:2]
        (padding_h, padding_w) = (int(H * padding), int(W * padding))
        # convert to grayscale
        # adjusted = cv2.convertScaleAbs(img.numpy(), alpha=1.5)
        gray = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2GRAY)

        # threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        avg_color_per_row = np.average(thresh, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        if avg_color > 255 / 2:
            # invert
            thresh = 255 - thresh

        # apply horizontal morphology close
        kernel = np.ones((5 , 100), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # get external contours
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # extract tensor patches
        patches = []
        # convert to bounding box coordinates and sort by row
        bboxes = list(map(cv2.boundingRect, contours))
        for bbox in bboxes:
            x,y,w,h = bbox
            hs = max(0, y-padding_h)
            he = min(y+h+padding_h, H)

            ws = max(0, x-padding_w)
            we = min(x+w+padding_w, W)

            patch= img[hs:he, ws:we, :]
            patches.append(patch)
        
        return patches



    