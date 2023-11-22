import tensorflow as tf
from PIL import Image
import numpy as np

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Returns an float list from a bool / enum / int / uint."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(values, keys, fs, d, limit=64):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {}
    for value, key, f in zip(values, keys, fs):
        if key != "img":
            paddings = tf.constant([[0, limit - len(value)]])
            value = tf.pad(value, paddings, constant_values=d["<unk>"])
            tensor = f(value)
            feature[key] = tensor
        else:
            feature[key] = f(value)

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def getImage(img, height=224, width=224):
    tensor = tf.constant(tf.keras.utils.img_to_array(img), dtype=tf.uint8)
    img_tensor = tf.cast(tf.image.resize_with_pad(tensor, height, width), dtype=tf.uint8)
    return tf.io.encode_png(img_tensor)

def getImageGrey(img, height=224, width=224):
    tensor = tf.constant(tf.keras.utils.img_to_array(img))
    tensor = tf.image.rgb_to_grayscale(tensor)
    img_tensor = tf.image.resize_with_pad(tensor, target_height=height, target_width=width)
    return img_tensor[tf.newaxis, :]

# the average height and width of the images is around 4000x200
def write_record_patches(entries, filename, patch_shape=(5, 5)):
  
    with tf.io.TFRecordWriter(filename) as writer:
        for entry in entries:
            img = Image.open(entry["img"])
            input = entry["input"]
            label = entry["label"]
            try:
                timg = getImageGrey(img, height=10, width=400)
            except:
                print(img.height, img.width)
                continue
            # extract 10x10 patches from the image as input sequence
            patches = tf.image.extract_patches(timg, 
                                               sizes=[1, patch_shape[0], patch_shape[1], 1], 
                                               strides=[1, patch_shape[0], patch_shape[1], 1], 
                                               rates=[1, 1, 1, 1], 
                                               padding="VALID")
            # squeeze to remove extra dim
            values = [input, label, tf.io.serialize_tensor(tf.squeeze(patches))]
            fs = [_int64_feature, _int64_feature, _bytes_feature]
            keys = ["input", "label", "img"]
            example = serialize_example(values, keys, fs)
            writer.write(example)

def write_record(entries, filename, d):
  
    with tf.io.TFRecordWriter(filename) as writer:
        for entry in entries:
            img = Image.open(entry["img"])
            input = entry["input"]
            label = entry["label"]
            try:
                timg = getImage(img, height=100, width=2000)
            except:
                print(entry)
                continue
            values = [input, label, timg]
            fs = [_int64_feature, _int64_feature, _bytes_feature]
            keys = ["input", "label", "img"]
            example = serialize_example(values, keys, fs, d)
            writer.write(example)
