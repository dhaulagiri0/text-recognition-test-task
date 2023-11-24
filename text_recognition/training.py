import tensorflow as tf
from ocr_model import OCRModel
from embedding_utils import load_json_data, load_embedding_json
from custom_layers import TokenOutput
from dataset import prepare_dataset, _parse_function
import datetime

# losses and acc matrices defined by tensorflow for transformer models
def masked_loss(labels, preds):  
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

    mask = (labels != 0) & (loss < 1e8) 
    mask = tf.cast(mask, loss.dtype)

    loss = loss*mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def masked_acc(labels, preds):
    mask = tf.cast(labels!=0, tf.float32)
    preds = tf.argmax(preds, axis=-1)
    labels = tf.cast(labels, tf.int64)
    match = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(match*mask)/tf.reduce_sum(mask)
    return acc

class GenerateText(tf.keras.callbacks.Callback):
    def __init__(self):
        self.image = "data/public_data/1.png"

    def on_epoch_end(self, epochs=None, logs=None):
        print()
        print()
        for t in (0.0, 0.5, 1.0):
            result = self.model.recognize_text(self.image, temperature=t)
            print(result)
        print()


def train_model(train_ds_path, valid_ds_path, vocab_size=409094, dictionary_path="dataset/vocab_bpemb.json"):
    
    patch_shape = [25, 25]
    train_ds = tf.data.TFRecordDataset(train_ds_path).map(_parse_function)
    valid_ds = tf.data.TFRecordDataset(valid_ds_path).map(_parse_function)

    train_ds = prepare_dataset(train_ds, batch_size=8)
    valid_ds = prepare_dataset(valid_ds, batch_size=8)

    checkpoint_filepath = 'dataset/checkpoints/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_masked_acc',
        mode='max',
        save_best_only=True)


    log_dir = "dataset/logs/fits" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)

    callbacks = [
        tensorboard_callback,
        model_checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True)]
    
    vocab_dict = load_json_data(dictionary_path)
    output_layer = TokenOutput(vocab_size, vocab_dict)
    output_layer.adapt(train_ds.map(lambda inputs, labels: labels))
    
    # embedding_weights, dim = load_embedding_json("dataset/embedding.json")
    # print("loaded embedding weigths")

    model = OCRModel(output_layer, dictionary=vocab_dict, embedding_weights=None, image_shape=[100, 2000], patch_shape=patch_shape)
    # tf.keras.utils.plot_model(model, to_file="dataset/mode.png", show_shapes=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
           loss=masked_loss,
           metrics=[masked_acc])
    
    # for (inputs, ex_labels) in train_ds.take(1):

    #     model(inputs)

    # model.summary()

    history = model.fit(
        train_ds.repeat(),
        steps_per_epoch=1000,
        validation_data=valid_ds.repeat(),
        validation_steps=200,
        epochs=100,
        callbacks=callbacks)
    
if __name__ == '__main__':

    train_model("dataset/records/train.tfrecord", "dataset/records/valid.tfrecord")
