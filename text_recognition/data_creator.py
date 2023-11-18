
"""
Current plan is to use some English dataset, maybe Maven, 
and translate some of the texts to Chinese using python's translate library

It is possible to translate each sentence fully or partially, 
this would increase the amount of data available but perhaps at the 
expense of the quality of the chinese dataset

We can then generate images based on sentences using 
"""
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from bpemb import BPEmb
import translators as ts
import json, io, os
import random
from tqdm import tqdm

def make_image(text, fontsize=10, padding=10, image_background=(255, 255, 255), text_color=(0, 0, 0)):

    font = ImageFont.truetype("dataset/NotoSansSC-VariableFont_wght.ttf", fontsize)

    img = Image.new('RGB', (1, 1), color = image_background)
    d = ImageDraw.Draw(img)
    d.text((0, 0), text, font=font, fill=text_color)
    bbox = d.textbbox((0, 0), text, font=font)
    d.rectangle(bbox, outline="red")

    x1, y1, x2, y2 = bbox
    image_width = x2 - x1 + padding
    image_height = y2 - y1 + padding

    img2 = Image.new('RGB', (image_width, image_height*2), color = image_background)
    d = ImageDraw.Draw(img2)
    d.text((0, 0), text, font=font, fill=text_color)
    # text_height = d.fontsize('Hello')

    # img = Image.new('RGB', (200, 100), (255, 255, 255))

    return img2

#randomly sample from two strings to create one string
# rate determines the percentage of an English sentence being translated to Chinese
def mix_trans(en, rate):
    mixed = []
    next = ""
    for wen in en.split(" "):
        next = ts.translate_text(wen, to_language='chi') if random.uniform(0, 1) <= rate else wen 
        mixed.append(next)
    return " ".join(mixed)

def mix_translate(sentences, rate=0.3):
    # translator= Translator(to_lang="zh")
    for d in tqdm(sentences):
        sentence = d["sentence"]
        if "translated" not in d and len(sentence) < 1000:
            full = ts.translate_text(sentence, to_language='chi')
            # full = translator.translate(sentence)
            d["mixed"]      = mix_trans(sentence, rate)
            d["translated"] = full

# dealing with the MAVEN dataset
def load_maven(path="dataset", dataset="train"):
    sentences = []
    # multibpemb = BPEmb(lang="multi", vs=1000000, dim=300)
    with open(os.path.join(path, f"{dataset}.jsonl")) as train_file:
        total = 0
        for i, line in enumerate(train_file):
            total = max(total, i)
            # I only need cur_json["content"], these are the sentences in the wiki page
            cur_json = json.loads(line)
            for sent_dict in cur_json["content"]:
                sentence = sent_dict["sentence"]
                sentences.append({"sentence": sentence})
                # ids, embedding = get_embedding_pair(sentence, multibpemb)
                # train_sentences.append({
                #     "sentence": sentence,
                #     "ids": ids,
                #     "embedding": embedding 
                # })
        print(f"loaded {total} articles for {dataset}. Totalling {len(sentences)} sentences")
    return sentences


def get_words_dict(entries, d, return_unknown=False):
    id = 0
    unknowns = []
    for i, entry in enumerate(entries):
        en = entry["sentence"]
        chi = entry["translated"] if "translated" in entry else []
        for char in en:
            if char not in d:
                d[char] = id
                id += 1
                if return_unknown:
                    unknowns.append(i)
        for char in chi:
            if char not in d:
                d[char] = id
                id += 1
                if return_unknown:
                    unknowns.append(i)
    d["<START>"] = id + 1
    d["<END>"] = id + 2
    d["<UNK>"] = id + 3
    return unknowns

def make_vocab(entries, path="dataset/vocab.json"):
    d = {}
    get_words_dict(entries, d)
    with open(path, 'w') as j:
        json.dump(d, j)
