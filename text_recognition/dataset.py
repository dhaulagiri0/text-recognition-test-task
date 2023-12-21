import tensorflow as tf

feature_description = {
    'img': tf.io.FixedLenFeature([], tf.string),
    'input': tf.io.FixedLenFeature([32], tf.int64),
    'label': tf.io.FixedLenFeature([32], tf.int64)
}

def detokenise(tokens, d):
    sentence = ""
    for token in tokens:
        if token != 0:
            word = d[token]
            if word not in ["<s>", "</s>"]:
                sentence += word
    return sentence.replace("▁", " ")

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def getFeatureExtractor():
    IMAGE_SHAPE=(224, 224, 3)

    mobilenet = tf.keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        include_preprocessing=True)
    mobilenet.trainable=False

    return mobilenet


def prepare_dataset(ds, batch_size=32, shuffle_buffer=1000):

    # trainAug = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.RandomBrightness(factor=[-0.5, 0.5]),
    #         tf.keras.layers.RandomContrast(factor=0.5)
    #     ]
    # )
    

    def prepare(sample):
        img = tf.io.decode_png(sample["img"])
        img = tf.image.grayscale_to_rgb(img) 
        img = tf.cast(img, tf.float32) # * tf.constant(1/255.) 
        input_tokens = sample["input"]
        label_tokens = sample["label"]
        return (img, input_tokens), label_tokens
    
    # def getpatches(inputs, label):
    #     img, input = inputs
    #     if img.shape[-1] != 1: 
    #         tensor = tf.image.rgb_to_grayscale(img)
    #     tensor = tf.cast(tensor, tf.float32) * tf.constant(1/255.) 
    #     # tensor = tensor[tf.newaxis, :]
    #     # patches = tf.image.extract_patches(tensor, 
    #     #                             sizes=[1, patch_shape[0], patch_shape[1], 1], 
    #     #                             strides=[1, patch_shape[0], patch_shape[1], 1], 
    #     #                             rates=[1, 1, 1, 1], 
    #     #                             padding="VALID")
    #     # patches = tf.ensure_shape(patches, (1, 4, 80, 625))
    #     # patches = tf.reshape(patches, (patches.shape[0], patches.shape[1] * patches.shape[2], patches.shape[-1]))
    #     # patches = tf.squeeze(patches)
    #     print(tensor.shape)
    #     return (tensor, input), label


    # Load the images and make batches.
    ds = (ds
            .shuffle(10000)
            .map(prepare, tf.data.AUTOTUNE)
            # .map(lambda x, y: ((trainAug(x[0]), x[1]), y), tf.data.AUTOTUNE)
            .map(lambda x, y: ((tf.math.scalar_mul(1/255., x[0]), x[1]), y), tf.data.AUTOTUNE)
            .map(lambda x, y: ((tf.ensure_shape(x[0], (150, 450, 3)), x[1]), y), tf.data.AUTOTUNE)
            .apply(tf.data.experimental.ignore_errors())
            .batch(batch_size))

    return (ds
            .unbatch()
            .shuffle(shuffle_buffer)
            .batch(batch_size)
            )


# def prepare_dataset_(ds, featureExtractor, batch_size=32, shuffle_buffer=1000):

#     trainAug = tf.keras.Sequential(
#         [
#             tf.keras.layers.RandomBrightness(factor=[-0.5, 0.5]),
#             tf.keras.layers.RandomContrast(factor=0.5),
#             # tf.keras.layers.RandomRotation(factor=[-0.1, 0.1]),
#         ]
#     )
    

#     def prepare(sample):
#         img = tf.io.decode_png(sample["img"])
#         input_tokens = sample["input"]
#         label_tokens = sample["label"]
#         tf.ensure_shape(img, 1, 100, 2000, 1)
#         return (img, input_tokens), label_tokens

#     def getFeature(pair, label):
#         img, input = pair
#         test_img_batch = tf.keras.applications.mobilenet_v3.preprocess_input(img[tf.newaxis, :])
#         return (tf.squeeze(featureExtractor(test_img_batch)), input), label

#     def to_tensor(inputs, labels):
#         (images, in_tok), out_tok = inputs, labels
#         return (images, in_tok), out_tok


#     # Load the images and make batches.
#     ds = (ds
#             .shuffle(10000)
#             .map(prepare, tf.data.AUTOTUNE)
#             .map(lambda x, y: ((trainAug(x[0]), x[1]), y), tf.data.AUTOTUNE)
#             .map(getFeature, tf.data.AUTOTUNE)
#             .apply(tf.data.experimental.ignore_errors())
#             .batch(batch_size))

#     return (ds
#             .unbatch()
#             .shuffle(shuffle_buffer)
#             .batch(batch_size)
#             .map(to_tensor, tf.data.AUTOTUNE)
#             )