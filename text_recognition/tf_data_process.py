import tensorflow as tf
from PIL import Image
import numpy as np

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(values, keys, fs, limit=64):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {}
    for value, key, f in zip(values, keys, fs):
        if key != "img":
            paddings = tf.constant([[0, limit - len(value)]])
            value = tf.pad(value, paddings)
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


def write_record(entries, filename, limit=64):
  
    with tf.io.TFRecordWriter(filename) as writer:
        for entry in entries:
            img = Image.open(entry["img"])
            input = entry["input"]
            label = entry["label"]
            try:
                timg = getImage(img)
            except:
                print(entry)
                continue
            values = [input, label, timg]
            fs = [_int64_feature, _int64_feature, _bytes_feature]
            keys = ["input", "label", "img"]
            example = serialize_example(values, keys, fs)
            writer.write(example)
