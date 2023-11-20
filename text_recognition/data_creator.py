
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

def load_json_data(path="dataset/raw/test_slid.json"):
    f = open(path)
    l = json.load(f)
    return l   

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

# eatch data entry contains path to image file, zs, label(tokens shifted by one position)
def create_dataset(json_path, set="train", img_path="dataset/images/", fontsize=70, padding=10, image_background=(255, 255, 255), text_color=(0, 0, 0), createImage=True):
    entries = load_json_data(f"{json_path}{set}.json")
    imgs = {}
    data = []
    for i, entry in enumerate(entries):
        original = entry["original"]
        tokens = entry["tokens"]
        id = i
        if createImage and (original not in imgs):
            imgs[original] = id
            img = make_image(original, fontsize, padding, image_background, text_color)
            img.save(f"{img_path}{set}/{str(i)}.png")
        elif original in imgs:
            id = imgs[original]
        else:
            imgs[original] = id

        input = tokens[:-1]
        output = tokens[1:]
        data.append({
            "img": f"{img_path}{set}/{str(id)}.png",
            "input": input,
            "label": output
        })
    return data
        


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

# the dataset obtained does not contain curly braces or new lines, we I am manually inserting some of these characters
def insert_char(sentences, characters="{}\n"):
    rate = 0.05
    for sentence in sentences:
        en = sentence["sentence"]
        chi = sentence["translated"] if "translated" in sentence else []
        mix = sentence["mixed"] if "mixed" in sentence else []
        if random.uniform(0, 1) <= rate:
            for c in characters:
                n = random.randint(1, len(en) - 1)
                en1, en2 = en[0:n], en[n:]
                en = sentence["sentence"] = f"{en1} {c} {en2}"

                if chi != []:
                    n = random.randint(1, len(chi) - 1)
                    chi1, chi2 = chi[0:n], chi[n:]
                    chi = sentence["translated"] = f"{chi1} {c} {chi2}"

                    n = random.randint(1, len(mix) - 1)
                    mix1, mix2 = mix[0:n], mix[n:]
                    mix = sentence["mixed"] = f"{mix1} {c} {mix2}"


def make_vocab(entries, path="dataset/vocab.json"):
    d = {}
    get_words_dict(entries, d)
    with open(path, 'w') as j:
        json.dump(d, j)

def load_json_entries(path="dataset/raw/", dataset="train"):
    f = open(path + dataset)
    l = json.load(f)
    return l     

def check_piece(pieces, d):
    ps = []
    for piece in pieces:
        if piece not in d:
            for i in piece: ps.append(i) if i in d else ps.append("<unk>")
        else:
            ps.append(piece)
    return ps

# takes in a list of setence pieces, corresponding token
# return two lists
def slide_sentence(pieces, tokens, limit=64, window = 5):
    if len(pieces) <= limit:
        return [pieces], [tokens]
    
    new_t = []
    new_p = []
    steps, remainder = divmod((len(pieces) - limit), window)
    for i in range(1, steps + 1):
        start = i*5
        end = i*5 + limit

        sub_p = pieces[start:end]
        sub_t = tokens[start:end]

        new_t.append(sub_t)
        new_p.append(sub_p)
    if remainder != 0:
        new_p.append(pieces[(len(pieces) - limit):])
        new_t.append(tokens[(len(pieces) - limit):])

    return new_p, new_t



# process all sentences, add BOS and EOS, split sequences longer than 512
def process_raw_entries(entries, sp, d):
    processed = []
    for entry in entries:
        en = entry["sentence"]
        chi = []
        mix = []
        if "translated" in entry:
            chi = entry["translated"]
            mix = entry["mixed"]
        for sentence in [en, chi, mix]:
            if sentence != []:
                #a adds BOS and EOS
                pieces = ["<s>"] + sp.EncodeAsPieces(sentence) + ["</s>"]
                piecess = check_piece(pieces, d)
                processed.append({
                    "original" : sentence,
                    "sequence" : piecess,
                    "tokens": [d[i] for i in piecess]
                })
    return processed

def process_limited_entries(entries, limit=64, window=5):
    new = []
    for entry in entries:
        if len(entry["sequence"]) > limit:
            ps, ts = slide_sentence(entry["sequence"], entry["tokens"], limit, window)
            for p, t in zip(ps, ts):
                new.append({
                    "original": entry["original"],
                    "sequence": p,
                    "tokens": t
                })
        else:
            new.append({
                "original": entry["original"],
                "sequence": entry["sequence"],
                "tokens": entry["tokens"]
            })
    return new