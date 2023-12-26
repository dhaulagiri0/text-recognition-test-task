# OCR Task

## Overview
Although I managed to implement a transformer OCR model in tensorflow, I was
unable to train it sufficiently to achieve a satisfactory performance. 

I have tried many different versions of the base transformer model, from altering
the training data to modifying the moder architecture and hyper parameters, none
was able to converge properly on my dataset. On hindsight, trying to have a single
model recognise both Chinese and English text at once 
was overly ambitious.

The final solution included in this repository simply uses a model from [easyOCR](https://pypi.org/project/easyocr/) to
detect Chinese and English sentences in an Image. This model uses a [DBNet](https://arxiv.org/abs/1911.08947) to perform
segmentation on the image to identify the location of text blobs within a image before
applying separate OCR models to detect text in different languages. This divided and
conquer approach might be the way to go for multilingual OCR.

## Process
### Data Creation
I began with implementing my own model pretty much from scratch using Tensorflow and training on 
a dataset consisting of purely English, purely Chinese and mixed English and Chinese sentences. These sentences
were generated from the MAVEN dataset. 

Although MAVEN consists of purely English Wikipedia articles, I used the python [translators](https://pypi.org/project/translators/) library
to fully translate some of the sentences and created mixed sentences by mixing purely English
and purely Chinese sentences. Since some of the sentences can get quite long, I processed the sentences
again to create sentence pieces that are at max 8 words in length.

I then created different variations of the dataset, from tokenising sentences based on characters to using
[sentence piece](https://github.com/google/sentencepiece) to separate each sentence into subwords tokens. Either way,
each token sequence has a maximum length of 32. For character based tokenisation, the dictionary consisted 4686 unique
characters while for sentence piece tokenisation, the dictionary consisted of 10000 unique words.

I then generated images for each short sentence in the dataset, varying the text and background colour while ensuring 
readability. Each image is then padded to 75x450 without changing the aspect ratio. 

### Model
I used a transformer as my base model architecture as it is an architecture that has shown
great promise in the field of NLP in the recent years. Further, it also seemed like a good
opportunity to try and learn about a new model architecture that I have never tried using before,

The transformer model consists of an encoder and decoder. The decoder in this case is what you would
expect for a normal transformer decoder for text data. My model consists of 12 decoder layers, each consisting
of a self attention layer into a cross-attention layer into a feed forward layer. Each decoder layer
has 6 attention heads. Before the text data is fed into the decoder, it is processed by a text embedding layer using
embeddings provided by [BPEMB](https://bpemb.h-its.org/multi/). The BPEMB multi is a pretrained embedding
for 275 languages, I processed this to include only tokens that appeared in my dataset.

For the encoder, I tried a few variations. One variation is very similar to that found in the [TrOCR model](https://huggingface.co/docs/transformers/model_doc/trocr)
Extracting 25x25 patches from the input image before linearly projecting each patch using a Dense layer as a form of embedding.
The other method involved using a MobilenetV3 Small pretrained on imagenet for feature extraction before projecting
with a Dense layer.

For both the encoder and decoder embedding, the embedding outputs undergo positional encoding similar to your typical
transformer models.

### Training
I trained the model primarily on my desktop which had a decent enough GPU, though not nearly powerful enough
to run iterations with batch sizes larger than 8. Generally, the model tend to stagnate at around 20% accuracy, constantly
outputting common characters such as space or fullstops.

### Predicting
Given that I have split each sentence in my training data into smaller pieces, the model would only be able to predict
shorter sequences of text. I used a simple algorithm utilising cv2's image thresholding, morphological transformation and contouring
to detect text blob and extracting sentence pieces from a given image. This is the only part of the project that 
seemed to have worked reasonably well.

## Afterthought
Given that the model that I built doesn't really work, there's many different things that I would like to try if 
given more time (I would probably try them anyways in my own time). 

One would be to actually perform a proper dataset analysis
before jumping into the project straight on. The entire data creation process was quite a mess as I was just running bits of my code 
in a Jupyter notebook (not included in this repo) to get everything to work.

Another thing that I would like to try is separating Chinese and English data and training separate transformer models.
This would necessitate the use of another network to detect text blobs and categorise them based on language.

Generally speaking, I could have carried this project out much more methodically.