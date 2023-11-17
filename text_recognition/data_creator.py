
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
import io

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

img = make_image(u"Some English and Chinese 这是一个中文句子", 240)
img.save("123.png")