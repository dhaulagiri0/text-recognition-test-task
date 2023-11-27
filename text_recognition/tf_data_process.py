import tensorflow as tf
from PIL import Image
import json, tqdm

"""
Generates tfrecords from json files containing
short_data

if provided with record_path pointing to an existing
tfrecord file, the processor automatically loads that record
target_img_shape and max_sequence_length should also be specified

record_path needs to be directly pointing to a tfrecord file

to prepare tfrecord for training, run prepare_dataset
"""
class TfDataProcessor():

    def __init__(self,
                 dataset_name        = "train",
                 short_data_path     = "dataset/short_json",
                 dict_path           = "dataset/vocab.json",
                 output_path         = "dataset/records_short",
                 max_sequence_length = 32,
                 target_img_shape    = [150, 450],
                 record_path         = None):
        
        self.dataset_name        = dataset_name
        self.target_img_shape    = target_img_shape
        self.max_sequence_length = max_sequence_length

        if record_path:
            self.record_path     = f"{record_path}/{dataset_name}.tfrecord"
        else:
            self.record_path     = None
            self.output_path     = output_path
            self.short_data      = self._load_json_data(f"{short_data_path}/{dataset_name}.json")
            self.word_2_token    = self._load_json_data(dict_path)

        self.feature_description = {
            'img': tf.io.FixedLenFeature([], tf.string),
            'input': tf.io.FixedLenFeature([max_sequence_length], tf.int64),
            'label': tf.io.FixedLenFeature([max_sequence_length], tf.int64)
        }

    # simple method to laod a json file
    @staticmethod
    def _load_json_data(path):
        f = open(path)
        l = json.load(f)
        return l   
    
    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(self, values, keys, fs):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {}
        for value, key, f in zip(values, keys, fs):
            # pads the text input and label to 32 characters
            if key != "img":
                paddings = tf.constant([[0, self.max_sequence_length - len(value)]])
                value = tf.pad(value, paddings, constant_values=0)
                tensor = f(value)
                feature[key] = tensor
            else:
                # serialise image
                feature[key] = f(value)

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    """
    Pads an image such that it is positioned at the top left of 
    the output

    image should be a PIL image
    """
    @staticmethod
    def add_padding(image, target_shape=[150, 450]):

        tensor = tf.keras.utils.img_to_array(image)

        if len(tensor.shape) != 4:
            tensor = tensor[tf.newaxis, :]
        
        height = tensor.shape[1]
        width = tensor.shape[2]
        target_height = target_shape[0]
        target_width = target_shape[1]

        height_ratio = (target_height / height)
        width_ratio = (target_width / width)
        
        if int(width_ratio * height) > target_shape[0]: # expand height to target
            final_width = int(height_ratio * width)
            tensor = tf.image.resize_with_pad(tensor, target_height, final_width)
            paddings = tf.constant([[0, 0], [0, 0], [0, target_shape[1] - final_width], [0, 0]])
        else: # expand width tp target
            final_height = int(width_ratio * height)
            tensor = tf.image.resize_with_pad(tensor, final_height, target_width)
            paddings = tf.constant([[0, 0], [0, target_shape[0] - final_height], [0, 0], [0, 0]])
        
        tensor = tf.pad(tensor, paddings)

        return tensor
    

    def write_record_short(self):

        d = self.word_2_token
        
        with tf.io.TFRecordWriter(f"{self.output_path}/{self.dataset_name}.tfrecord") as writer:
            cnt = 0
            for entry in tqdm.tqdm(self.short_data):
                if len(entry["tokens"]) > (self.max_sequence_length - 2): continue

                img = Image.open(entry["img"])
                tokens = [i + 1 for i in entry["tokens"]]
                input = [d["<s>"]] + tokens
                label = tokens + [d["</s>"]]

                timg = self.add_padding(img, target_shape=self.target_img_shape)[0]
                timg = tf.image.rgb_to_grayscale(timg)

                # cast to uint then encode
                # encoded image is a string
                timg = tf.io.encode_png(tf.cast(timg , dtype=tf.uint8))

                values = [input, label, timg]
                fs = [self._int64_feature, self._int64_feature, self._bytes_feature]
                keys = ["input", "label", "img"]

                example = self.serialize_example(values, keys, fs)
                writer.write(example)

                cnt += 1

            print(f"Serialised {cnt} samples")
            self.record_path = f"{self.output_path}/{self.dataset_name}.tfrecord"


    def _parse_function(self, example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, self.feature_description)
    
    """
    Prepares the tfrecord for data generation
    the dataset object returned can be used directly passed 
    to model.fit
    """
    def prepare_dataset(self, batch_size=32, shuffle_buffer=1000):
        
        # some basic image augmentation
        trainAug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomBrightness(factor=[-0.5, 0.5]),
                tf.keras.layers.RandomContrast(factor=0.5)
            ]
        )

        def prepare(sample):
            img = tf.io.decode_png(sample["img"])
            img = tf.cast(img, tf.float32) # * tf.constant(1/255.) 
            input_tokens = sample["input"]
            label_tokens = sample["label"]
            return (img, input_tokens), label_tokens
        
        if self.record_path:
            ds = tf.data.TFRecordDataset(self.record_path).map(self._parse_function)

            # Load the images and make batches.
            ds = (ds
                    .shuffle(10000)
                    .map(prepare, tf.data.AUTOTUNE)
                    .map(lambda x, y: ((trainAug(x[0]), x[1]), y), tf.data.AUTOTUNE)
                    .map(lambda x, y: ((tf.math.scalar_mul(1/255., x[0]), x[1]), y), tf.data.AUTOTUNE)
                    .map(lambda x, y: ((tf.ensure_shape(x[0], self.target_img_shape + [1]), x[1]), y),
                                        tf.data.AUTOTUNE)
                    .batch(batch_size))

            return (ds
                    .unbatch()
                    .shuffle(shuffle_buffer)
                    .batch(batch_size)
                    )
        
        else:
            print("path to tfrecord file not provided. Try writing tfrecord first.")
            return None
        



### ------------- legacy functions ------------- ###




def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

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

def add_padding(image, target_shape=[100, 2000], encode=True):
    tensor = tf.keras.utils.img_to_array(image)
    if len(tensor.shape) != 4:
        tensor = tensor[tf.newaxis, :]
    height = tensor.shape[1]
    width = tensor.shape[2]
    target_height = target_shape[0]
    target_width = target_shape[1]

    height_ratio = (target_height / height)
    width_ratio = (target_width / width)

    if int(width_ratio * height) > target_shape[0]: # resize height to 100
        final_width = int(height_ratio * width)
        tensor = tf.image.resize_with_pad(tensor, target_height, final_width)
        paddings = tf.constant([[0, 0], [0, 0], [0, target_shape[1] - final_width], [0, 0]])
    else: # resize width
        final_height = int(width_ratio * height)
        tensor = tf.image.resize_with_pad(tensor, final_height, target_width)
        paddings = tf.constant([[0, 0], [0, target_shape[0] - final_height], [0, 0], [0, 0]])
    
    tensor = tf.image.rgb_to_grayscale(tf.pad(tensor, paddings))

    if encode: return tf.io.encode_png(tf.cast(tensor , dtype=tf.uint8))
    else: return tensor

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
            # try:
            #timg = getImage(img, height=100, width=2000)
            timg = add_padding(img, target_shape=[100, 2000])[0]
            # except:
            #     print(entry)
            #     continue
            values = [input, label, timg]
            fs = [_int64_feature, _int64_feature, _bytes_feature]
            keys = ["input", "label", "img"]
            example = serialize_example(values, keys, fs, d)
            writer.write(example)

def write_record_short(entries, filename, d):
  
    with tf.io.TFRecordWriter(filename) as writer:
        cnt = 0
        for entry in entries:
            if len(entry["tokens"]) > 30: continue

            img = Image.open(entry["img"])
            input = [31] + entry["tokens"]
            label = entry["tokens"] + [30]
            # if not re.search(u'[\u4e00-\u9fff]', input) and random.randint(0, 10) > 4: 
            #     continue
            # try:
            #timg = getImage(img, height=100, width=2000)
            timg = add_padding(img, target_shape=[150, 450])[0]
            # except:
            #     print(entry)
            #     continue
            values = [input, label, timg]
            fs = [_int64_feature, _int64_feature, _bytes_feature]
            keys = ["input", "label", "img"]
            example = serialize_example(values, keys, fs, d, limit=32)
            writer.write(example)

            cnt += 1
