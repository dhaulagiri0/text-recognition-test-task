from pathlib import Path
import tensorflow as tf
from custom_layers import DecoderLayer, SeqEmbedding, ImageEmbedding, EncoderLayer
from dataset import detokenise
from tf_data_process import TfDataProcessor
from PIL import Image
from pytesseract import Output
import easyocr, pytesseract
import cv2
import numpy as np


class OCRModel(tf.keras.Model):
    
    def __init__(self, 
                 use_reader=True,
                 mixed = False,
                 output_layer=None,
                 dictionary=None, 
                 embedding_weights=None,
                 image_shape=[150, 450], # HxW
                 patch_shape=[10, 10],
                 vocab_size=10000, 
                 num_heads=8,
                 num_layers=4, 
                 units=512,
                 max_length=32, 
                 dropout_rate=0.5):
        
        super().__init__()

        if use_reader: # library made model
            print("using reader")
            # sets up an ocr model using the easyocr lib
            # using cpu here since the GPU option somehow gets stuck on my machine
            if mixed:
                self.reader_en  = easyocr.Reader(['en'], gpu=False)
                self.reader_chi = easyocr.Reader(['ch_sim'], gpu=False)
            else:
                self.reader     = easyocr.Reader(['ch_sim','en'], gpu=False)
            print("model loaded")

            self.mixed = mixed
            
        else: # self made model
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
    
    # expects image to be a path string
    @staticmethod
    def img_preprocess(image: str, image_shape):

        # padding
        timg = TfDataProcessor.add_padding(image, target_shape=image_shape)
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
        
        return tokens
    
    # recognise text with separate ocr models for chinese and english
    def multiShot(self, image: Path):
        reader_en = self.reader_en
        reader_chi = self.reader_chi

        image = cv2.imread(image.absolute().as_posix())
        ps, bbs = OCRModel.getSubImages(image)

        result = []
        for p, bb in zip(ps, bbs):
            en = reader_en.readtext(np.asarray(p))
            chi = reader_chi.readtext(np.asarray(p))

            # offset bbox of each word
            if (chi == [] and en != []) or (en !=[] and en[0][-1] > chi[0][-1]):
                result += [(bb, en[0][1], en[0][2])]

            if (en == [] and chi != []) or (chi != [] and en[0][-1] < chi[0][-1]):
                result += [(bb, chi[0][1], chi[0][2])]

        return result
        
        
    # recognise text with easy ocr model, one shot
    def recognize_text(self, image: Path):

        if not self.mixed:
            result = self.reader.readtext(image.absolute().as_posix())
        else:
            result = self.multiShot(image)

        overall = ""
        
        if len(result) == 0: return overall

        if not self.mixed:
            last_y = 0
            last_x = 0
            avg_w = 0
            avg_h = 0

            # find average height and width of letters
            for r in result:
                print(r)
                bbox = r[0]

                x = bbox[0][0]
                y = bbox[0][1]
                w = bbox[1][0] - x
                h = bbox[2][1] - y

                avg_w += w // len(r[1])
                avg_h += h

            avg_h = avg_h // len(result)
            avg_w = avg_w // len(result)

            # construct the overall text
            # adds spaces and new lines depending on the location of the bboxes
            for i, r in enumerate(result):
                bbox = r[0]
                word = r[1]

                x = bbox[0][0]
                y = bbox[0][1]
                w = bbox[1][0] - x
                h = bbox[2][1] - y

                # if the current bit of text is significantly
                # lower than the previous, add new line
                if i != 0 and y - last_y >= avg_h * 0.6:
                    overall += "\n"

                # if the current bit of text is significantly
                # spaced out from the last bit, add space
                if i != 0 and x - last_x >= avg_w * 0.6:
                    overall += " "

                overall += f"{word}"
                
                last_y = y
                last_x = x
        else:
            prevLine = 0
            for r in result: 
                if r[0][0] != prevLine:
                    prevLine = r[0][0]
                    overall += "\n"
                if r[0][1] != 0:
                    overall += " "
                overall += r[1]

        print(overall)
        return overall

    def recognize_text_(self, image: Path, temperature=1, get_patches=False) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image: The path to the image file.
        :return: The recognized text from the image.
        """
        img = Image.open(image).convert('RGB')
        # divide a large image into text blobs if necessary
        if get_patches:
            image_list = OCRModel.getSubImages_(img)
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

        return txt_seq
    
    # uses pytesseract to get image bboxes for individual words
    def getSubImages(img: np.array):

        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['level'])
        ps = []
        positions = []
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            level = d['level'][i]
            line = d['line_num'][i]
            word_num = d['word_num'][i]
            patch = np.pad(img[y:y+h, x:x+w, :], ((5, 5), (12, 12), (0, 0)),'edge')

            # get word level
            if level == 5:
                ps.append(patch)
                positions.append((line, word_num))

        return ps, positions


    