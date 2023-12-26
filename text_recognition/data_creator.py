
"""
Current plan is to use some English dataset, maybe Maven, 
and translate some of the texts to Chinese using google/bing translate

It is possible to translate each sentence fully or partially, 
this would increase the amount of data available but perhaps at the 
expense of the quality of the chinese dataset

I can then generate images based on sentences using 
"""

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import translators as ts
import json, os
import random
import tqdm
import sentencepiece as spm

"""
A class to generate usable datasets from Maven
The data generated (short_data) is of the following format:
[
 {
    "img"      : path_to_image_data,
    "tokens"   : tokenised_sequence,
    "sequence" : raw_sequence (en, chi or mixed)
 }
]

An intermediary list, base, is also generated of the
following format:
[
 {
    "sentence"   : sentence_in_English,
    "translated" : sentence_in_Chinese,
    "mixed"      : sentence_with_chi_and_en
 }
]
The sequences in short_data are shorter pieces of texts 
extracted from the sentences in base

short_data needs to be further processed into tfrecords
for model training (using tf_data_process.py)
"""
class JsonDataCreator():

    def __init__(self,
                 maven_path="dataset/",
                 dataset_name="train",
                 translate_rate=.5,
                 vocab_path= "dataset/vocab.json",
                 img_path="dataset/short_imgs",
                 font_size=50,
                 font_dir="dataset/fonts",
                 short_data_path=None,
                 clean_path=None,
                 use_spm=True,
                 model_file="dataset/engchi.model"):
        
        self.maven_path=maven_path
        self.dataset_name=dataset_name
        self.img_path=img_path
        self.font_size=font_size
        self.font_dir=font_dir

        # this determine how much of an English sentence is translated
        self.translate_rate = translate_rate

        # get the base dataset
        # which is a list of sentences
        if not short_data_path and not clean_path:
            self.base = self.load_maven()


            # check if vocabulary json is provided
            # if not, make a new one
            # the vocab file should be a dictionary of
            # word to token
            if vocab_path and os.path.isfile(vocab_path):
                self.word_2_token = JsonDataCreator._load_json_data(vocab_path)
            else:
                if use_spm:
                    sp = spm.SentencePieceProcessor(model_file=str(model_file))
                    self.word_2_token ={sp.id_to_piece(id):id for id in range(sp.get_piece_size())}
                else:
                    self.word_2_token = self.create_vocab()

            self.vocab_size = len(self.word_2_token)

        elif short_data_path:
            self.short_data = JsonDataCreator._load_json_data(short_data_path)
        else:
            self.base =  JsonDataCreator._load_json_data(clean_path)
            self.word_2_token = JsonDataCreator._load_json_data(vocab_path)

    # simple method to laod a json file
    def _load_json_data(path):
        f = open(path)
        l = json.load(f)
        return l   

    # dealing with the MAVEN dataset
    def load_maven(self, export=False, output="dataset/"):

        path = self.maven_path
        dataset = self.dataset_name
        sentences = []
        chi = []
        en = []
        with open(f"{path}/{dataset}.jsonl") as f:
            total = 0
            for i, line in tqdm.tqdm(enumerate(f)):
                total = max(total, i)
                # I only need cur_dict["content"], these are the sentences in the wiki page
                cur_dict = json.loads(line)
                for sent_dict in cur_dict["content"]:
                    sentence = sent_dict["sentence"]
                    
                    # translate to chinese
                    translated = ts.translate_text(sentence, to_language='chi')
                    # mix
                    mixed      = self._mix_trans(sentence, self.translate_rate)

                    sentences.append({"sentence"  : sentence,
                                      "translated": translated,
                                      "mixed"     : mixed})
                    if export:
                        chi.append(translated)
                        en.append(sentence)
                if i > 2: break
            print(f"loaded {total} articles for {dataset}. Totalling {len(sentences)} sentences")

        if export:
            with open(f"{output}/chi.txt", "w") as f:
                for c in chi:
                    f.writelines(c + "\n")

            with open(f"{output}/en.txt", "w") as f:
                for c in en:
                    f.writelines(c + "\n")

        self.sentence = sentence
        return sentences

    
    """
     randomly sample from two strings to create one string
     rate determines the percentage of an English sentence being translated to Chinese
     I chose to translate words individually since simply mixing a pair of 
     English and Chinese sentence would destroy the meaning
     This operation is quite resource intensive
    """
    def _mix_trans(self, en, rate):

        def single_trans(word):
            if random.uniform(0, 1) <= rate:
                return ts.translate_text(word, to_language='chi')
            else: 
                return word

        mixed = en.split(" ")
        mixed = map(single_trans, mixed)
        return " ".join(mixed)
    
    def create_vocab(self):
        word_2_token = {}

        unique_en = ""
        unique_chi = ""

        for entry in self.base:
            en         = entry["sentence"]
            unique_en  = ''.join(set(en + unique_en))

            chi        = entry["translated"]
            unique_chi = ''.join(set(chi + unique_chi))

        token_id = 0
        for en in sorted(unique_en):
            word_2_token[en] = token_id
            token_id += 1
        
        for chi in sorted(unique_chi):
            word_2_token[chi] = token_id
            token_id += 1
        
        # add the special tokens
        word_2_token["<s>"]   = token_id
        word_2_token["</s>"]  = token_id + 1
        word_2_token["<unk>"] = token_id + 2
        word_2_token[" "]     = token_id + 3

        return word_2_token

    def add_vocab(self, word):
        if word not in self.word_2_token:
            self.word_2_token[word] = len(self.word_2_token)
    
    def tokenise(self, sequence, add_to_dict=False):

        word_2_token = self.word_2_token

        tokens = []
        for char in sequence:
            if char in word_2_token:
                token = word_2_token[char]
            else:
                # print(f"{char} not found in dictionary")
                if add_to_dict:
                    word_2_token[char] = token = len(word_2_token)
                else:
                    token = word_2_token["<unk>"]
            tokens.append(token)
            
        return tokens
    
    # exports an attribute as json
    def export(self, path, attribute):

        with open(path, "w") as f:
            json.dump(attribute, f)


    def make_image(self, text, fontsize=10, image_background=(255, 255, 255), text_color=(0, 0, 0)):

        fonts = os.listdir(self.font_dir)
        font = ImageFont.truetype(f"{self.font_dir}/{fonts[random.randint(0, len(fonts) - 1)]}", fontsize)

        # place holder image to get text size
        img = Image.new('RGB', (1, 1), color = image_background)
        d = ImageDraw.Draw(img)
        d.text((0, 0), text, font=font, fill=text_color)
        bbox = d.textbbox((0, 0), text, font=font)
        d.rectangle(bbox, outline="red")

        # caculate new image dimensions
        x1, y1, x2, y2 = bbox
        image_width    = x2 - x1 + 10
        image_height   = y2 - y1 + 10

        # adjust image height
        img2 = Image.new('RGB', (image_width, image_height*2), color = image_background)
        d    = ImageDraw.Draw(img2)
        d.text((0, 0), text, font=font, fill=text_color)

        return img2

    
    # generate short sequences from base
    """
    using shorter sequences allows the model to generate
    character sequences instead of words. This reduces
    vocab size, increases flexibility and also allow the 
    model to converge much faster
    """
    def gen_short(self, piece_length=8, create_imgs=True):

        base         = self.base
        img_path     = self.img_path

        short_data = []
        cnt = 0
        max_height = 0
        max_width = 0
        avg_height = 0
        avg_wdith = 0

        for entry in tqdm.tqdm(base):

            en    = entry["sentence"]
            en_tokens, en_pieces = self.split_sentence(en, piece_length)

            chi_tokens   = []
            chi_pieces   = []
            mixed_tokens = []
            mixed_pieces = []

            if "translated" in entry:
                chi   = entry["translated"]
                mixed = entry["mixed"]

                chi_tokens, chi_pieces     = self.split_sentence(chi, piece_length)
                mixed_tokens, mixed_pieces = self.split_sentence(mixed, piece_length)

            all_tokens = en_tokens + chi_tokens + mixed_tokens
            all_pieces = en_pieces + chi_pieces + mixed_pieces
            
            # zip tokens and pieces and remove potential repeats
            tokens_pieces = set([(" ".join(str(i) for i in k), v) for k ,v in zip(all_tokens, all_pieces)])

            for pair in tokens_pieces:

                # a dictionary to store data entries
                d = {}

                # generate random colours
                c = tuple(random.sample(range(0, 255), 3))

                # ensure there's white and dark background
                if random.randint(0, 10) > 3:
                    if random.randint(0, 10) > 3:
                        c = (255, 255, 255)
                    else:
                        c = (30, 30, 30)
                
                if create_imgs:
                    
                    img = self.make_image(pair[1], 
                                    self.font_size, 
                                    image_background=c, 
                                    text_color=tuple(255 - x for x in c))
                    
                    img.save(f"{img_path}/{self.dataset_name}/{cnt}.png")

                    avg_height += img.height
                    max_height = max(max_height, img.height)
                    avg_wdith  += img.width
                    max_width  = max(max_width, img.width)

                # assume image already exists
                d["img"]      = f"{img_path}/{self.dataset_name}/{cnt}.png"
                d["tokens"]   = [int(x) for x in pair[0].split(" ")]
                d["sequence"] = pair[1]

                short_data.append(d)
                cnt += 1

        if create_imgs:
            avg_wdith = avg_wdith // cnt
            avg_height = avg_height // cnt

            # some data about the images generated
            print(f"Average image dimens: {avg_height}x{avg_wdith}")
            print(f"Max image dimens:     {max_height}x{max_width}")
        
        self.short_data = short_data

    # split a sentence into pieces in word tokens
    # piece_length determines how long the pieces are
    def split_sentence(self, sentence, piece_length, add_to_dict=False):
        words = sentence.split(" ")
        end = len(words)

        tokens_list = []
        pieces_list = []
        # take generated_len number of words
        for i in range(0, end, piece_length):
            d = {}

            if i + piece_length < end:
                piece = words[i: i+piece_length]
            else:
                piece = words[i: end]

            piece = " ".join(piece)
            tokens_list.append(self.tokenise(piece, add_to_dict))
            pieces_list.append(piece)
        
        return tokens_list, pieces_list


# def getBbox(image):

    
#     image = cv2.imread(image)
#     orig = image.copy()
#     (H, W) = image.shape[:2]

#     (newW, newH) = (W - W%32, H - H%32)
#     # rW = W / float(newW)
#     # rH = H / float(newH)

#     image = cv2.resize(image, (newW, newH))
#     (H, W) = image.shape[:2]
#     print(H, W)

#     layerNames = [
# 	"feature_fusion/Conv_7/Sigmoid",
# 	"feature_fusion/concat_3"]

#     net = cv2.dnn.readNet("dataset/frozen_east_text_detection.pb")

#     blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
#         (123.68, 116.78, 103.94), swapRB=True, crop=False)
#     net.setInput(blob)
#     (scores, geometry) = net.forward(layerNames)
    
#     return geometry

### ------------- legacy functions ------------- ###


def load_json_data(path="dataset/raw/test_slid.json"):
    f = open(path)
    l = json.load(f)
    return l   

# def make_image(text, fontsize=10, padding=10, image_background=(255, 255, 255), text_color=(0, 0, 0)):

#     fonts = os.listdir("dataset/fonts/")
#     font = ImageFont.truetype(f"dataset/fonts/{fonts[random.randint(0, len(fonts) - 1)]}", fontsize)


#     img = Image.new('RGB', (1, 1), color = image_background)
#     d = ImageDraw.Draw(img)
#     d.text((0, 0), text, font=font, fill=text_color)
#     bbox = d.textbbox((0, 0), text, font=font)
#     d.rectangle(bbox, outline="red")

#     x1, y1, x2, y2 = bbox
#     image_width = x2 - x1 + padding
#     image_height = y2 - y1 + padding

#     img2 = Image.new('RGB', (image_width, image_height*2), color = image_background)
#     d = ImageDraw.Draw(img2)
#     d.text((0, 0), text, font=font, fill=text_color)
#     # text_height = d.fontsize('Hello')

#     # img = Image.new('RGB', (200, 100), (255, 255, 255))

#     return img2

# # eatch data entry contains path to image file, zs, label(tokens shifted by one position)
# def create_dataset(json_path, set="train", img_path="dataset/images/", fontsize=70, padding=10, image_background=(255, 255, 255), text_color=(0, 0, 0), createImage=True):
#     entries = load_json_data(f"{json_path}{set}.json")
#     imgs = {}
#     data = []
#     for i, entry in enumerate(entries):
#         original = entry["original"]
#         tokens = entry["tokens"]
#         id = i
#         if createImage and (original not in imgs):
#             imgs[original] = id
#             img = make_image(original, fontsize, padding, image_background, text_color)
#             img.save(f"{img_path}{set}/{str(i)}.png")
#         elif original in imgs:
#             id = imgs[original]
#         else:
#             imgs[original] = id

#         input = tokens[:-1]
#         output = tokens[1:]
#         data.append({
#             "img": f"{img_path}{set}/{str(id)}.png",
#             "input": input,
#             "label": output
#         })
#     return data
        

# #randomly sample from two strings to create one string
# # rate determines the percentage of an English sentence being translated to Chinese
# def mix_trans(en, rate):
#     mixed = []
#     next = ""
#     for wen in en.split(" "):
#         next = ts.translate_text(wen, to_language='chi') if random.uniform(0, 1) <= rate else wen 
#         mixed.append(next)
#     return " ".join(mixed)

# def mix_translate(sentences, rate=0.3):
#     # translator= Translator(to_lang="zh")
#     for d in tqdm(sentences):
#         sentence = d["sentence"]
#         if "translated" not in d and len(sentence) < 1000:
#             full = ts.translate_text(sentence, to_language='chi')
#             # full = translator.translate(sentence)
#             d["mixed"]      = mix_trans(sentence, rate)
#             d["translated"] = full

# # dealing with the MAVEN dataset
# def load_maven(path="dataset", dataset="train"):
#     sentences = []
    
#     with open(os.path.join(path, f"{dataset}.jsonl")) as train_file:
#         total = 0
#         for i, line in enumerate(train_file):
#             total = max(total, i)
#             # I only need cur_json["content"], these are the sentences in the wiki page
#             cur_json = json.loads(line)
#             for sent_dict in cur_json["content"]:
#                 sentence = sent_dict["sentence"]
#                 sentences.append({"sentence": sentence})
#                 # ids, embedding = get_embedding_pair(sentence, multibpemb)
#                 # train_sentences.append({
#                 #     "sentence": sentence,
#                 #     "ids": ids,
#                 #     "embedding": embedding 
#                 # })
#         print(f"loaded {total} articles for {dataset}. Totalling {len(sentences)} sentences")
#     return sentences


# def get_words_dict(entries, d, return_unknown=False):
#     id = 0
#     unknowns = []
#     for i, entry in enumerate(entries):
#         en = entry["sentence"]
#         chi = entry["translated"] if "translated" in entry else []
#         for char in en:
#             if char not in d:
#                 d[char] = id
#                 id += 1
#                 if return_unknown:
#                     unknowns.append(i)
#         for char in chi:
#             if char not in d:
#                 d[char] = id
#                 id += 1
#                 if return_unknown:
#                     unknowns.append(i)
#     d["<START>"] = id + 1
#     d["<END>"] = id + 2
#     d["<UNK>"] = id + 3
#     return unknowns

# # the dataset obtained does not contain curly braces or new lines, I am manually inserting some of these characters
# def insert_char(sentences, characters="{}\n"):
#     rate = 0.05
#     for sentence in sentences:
#         en = sentence["sentence"]
#         chi = sentence["translated"] if "translated" in sentence else []
#         mix = sentence["mixed"] if "mixed" in sentence else []
#         if random.uniform(0, 1) <= rate:
#             for c in characters:
#                 n = random.randint(1, len(en) - 1)
#                 en1, en2 = en[0:n], en[n:]
#                 en = sentence["sentence"] = f"{en1} {c} {en2}"

#                 if chi != []:
#                     n = random.randint(1, len(chi) - 1)
#                     chi1, chi2 = chi[0:n], chi[n:]
#                     chi = sentence["translated"] = f"{chi1} {c} {chi2}"

#                     n = random.randint(1, len(mix) - 1)
#                     mix1, mix2 = mix[0:n], mix[n:]
#                     mix = sentence["mixed"] = f"{mix1} {c} {mix2}"


# def make_vocab(entries, path="dataset/vocab.json"):
#     d = {}
#     get_words_dict(entries, d)
#     with open(path, 'w') as j:
#         json.dump(d, j)


# def check_piece(pieces, d):
#     ps = []
#     for piece in pieces:
#         if piece not in d:
#             for i in piece: ps.append(i) if i in d else ps.append("<unk>")
#         else:
#             ps.append(piece)
#     return ps

# # takes in a list of setence pieces, corresponding token
# # return two lists
# def slide_sentence(pieces, tokens, limit=64, window = 5):
#     if len(pieces) <= limit:
#         return [pieces], [tokens]
    
#     new_t = []
#     new_p = []
#     steps, remainder = divmod((len(pieces) - limit), window)
#     for i in range(1, steps + 1):
#         start = i*5
#         end = i*5 + limit

#         sub_p = pieces[start:end]
#         sub_t = tokens[start:end]

#         new_t.append(sub_t)
#         new_p.append(sub_p)
#     if remainder != 0:
#         new_p.append(pieces[(len(pieces) - limit):])
#         new_t.append(tokens[(len(pieces) - limit):])

#     return new_p, new_t


# # process all sentences, add BOS and EOS, split sequences longer than 512
# def process_raw_entries(entries, sp, d):
#     processed = []
#     for entry in entries:
#         en = entry["sentence"]
#         chi = []
#         mix = []
#         if "translated" in entry:
#             chi = entry["translated"]
#             mix = entry["mixed"]
#         for sentence in [en, chi, mix]:
#             if sentence != []:
#                 #a adds BOS and EOS
#                 pieces = ["<s>"] + sp.EncodeAsPieces(sentence) + ["</s>"]
#                 piecess = check_piece(pieces, d)
#                 processed.append({
#                     "original" : sentence,
#                     "sequence" : piecess,
#                     "tokens": [d[i] for i in piecess]
#                 })
#     return processed

# def process_limited_entries(entries, limit=64, window=5):
#     new = []
#     for entry in entries:
#         if len(entry["sequence"]) > limit:
#             ps, ts = slide_sentence(entry["sequence"], entry["tokens"], limit, window)
#             for p, t in zip(ps, ts):
#                 new.append({
#                     "original": entry["original"],
#                     "sequence": p,
#                     "tokens": t
#                 })
#         else:
#             new.append({
#                 "original": entry["original"],
#                 "sequence": entry["sequence"],
#                 "tokens": entry["tokens"]
#             })
#     return new

# def tokenise(sequence, word_token):
#     tokens = []
#     for char in sequence:
#         if char in word_token:
#             token = word_token[char]
#         else:
#             print(f"{char} not found in dictionary")
#             word_token[char] = token = len(word_token)
#             # token = word_token["<unk>"]
#         tokens.append(token)
#     return tokens, word_token

# # generate short sequences from raw json (around 5 words)
# def gen_short(json_path, output_path, word_token, img_path, generated_len=2):

#     entries = load_json_data(json_path)

#     new_sequences = []
#     cnt = 0
#     max_height = 0
#     max_width = 0
#     avg_height = 0
#     avg_wdith = 0
#     for entry in entries:
#         sentence = entry["original"]
#         words = sentence.split(" ")
#         end = len(words)

#         if not re.search(u'[\u4e00-\u9fff]', sentence) and random.randint(0, 10) > 4:
#             cnt += len(range(0, end, generated_len)) 
#             continue
#         # take generated_len number of words
#         for i in range(0, end, generated_len):
#             d = {}
#             if i + generated_len < end:
#                 sequence = words[i: i+generated_len]
#             else:
#                 sequence = words[i: end]
#             sequence = " ".join(sequence)
#             d["tokens"], word_token = tokenise(sequence, word_token)
#             d["sequence"] = sequence
#             c = tuple(random.sample(range(0, 255), 3))
#             if random.randint(0, 10) > 3:
#                 if random.randint(0, 10) > 3:
#                     c = (255, 255, 255)
#                 else:
#                     c = (30, 30, 30)
            
#             # if not os.path.isfile(f"{img_path}/{cnt}.png"):
#             img = make_image(sequence, 
#                             fontsize=70, 
#                             padding=10, 
#                             image_background=c, 
#                             text_color=tuple(255 - x for x in c))
#             img.save(f"{img_path}/{cnt}.png")
#             avg_height += img.height
#             max_height = max(max_height, img.height)
#             avg_wdith += img.width
#             max_width = max(max_width, img.width)
#             d["img"] = f"{img_path}/{cnt}.png"
#             new_sequences.append(d)

#             cnt += 1

#     avg_wdith = avg_wdith// cnt
#     avg_height = avg_height // cnt

#     print(avg_height, avg_wdith, max_height, max_width)

#     with open(output_path, "w") as f:
#         json.dump(new_sequences, f)
    
#     return new_sequences, word_token
    