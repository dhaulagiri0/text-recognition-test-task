import tensorflow as tf
from ocr_model import OCRModel
from embedding_utils import load_json_data, load_embedding_json
from custom_layers import TokenOutput
from dataset import prepare_dataset, _parse_function
import datetime
from PIL import Image
from tf_data_process import add_padding, TfDataProcessor
from dataset import detokenise

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
        self.image = "dataset/short_imgs/train/"

    def on_epoch_end(self, epochs=None, logs=None):
        print()
        print()
        for i in range(10):
            for t in (0.0, 0.5, 1.0):
                result = self.model.recognize_text(f"{self.image}/{i}.png", temperature=t)
                print(result)
            print()

def train_model(ds_path="dataset/records_short_new", 
                dictionary_path="dataset/vocab.json", 
                patch_shape = [25, 25], 
                image_shape=[150, 450], 
                batch_size=32, 
                weights=None):
    
    vocab_dict = load_json_data(dictionary_path)
    vocab_size = len(vocab_dict) + 1 # +1 for 0 padding

    train_ds = TfDataProcessor("train", record_path=ds_path)
    train_ds = train_ds.prepare_dataset(batch_size=batch_size)

    valid_ds = TfDataProcessor("valid", record_path=ds_path)
    valid_ds = valid_ds.prepare_dataset(batch_size=batch_size)
    
    # train_ds = tf.data.TFRecordDataset(train_ds_path).map(_parse_function)
    # valid_ds = tf.data.TFRecordDataset(valid_ds_path).map(_parse_function)
    # train_ds = prepare_dataset(train_ds, batch_size)
    # valid_ds = prepare_dataset(valid_ds, batch_size)

    checkpoint_filepath = 'dataset/checkpoints/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_masked_acc',
        mode='max',
        save_best_only=True)


    log_dir = "dataset/logs/fits" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False)

    callbacks = [
        tensorboard_callback,
        model_checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True),
        GenerateText(),]
    
    if weights:

        model = tf.keras.models.load_model(weights, custom_objects={
            "masked_acc"  : masked_acc,
            "masked_loss" : masked_loss
        })

        model.evaluate(valid_ds, batch_size = 32, steps = 20, verbose=2)
        # for i in range(0, 20):
        #     pred = recognize_text(model, vocab_dict, f"dataset/short_imgs/train/{i}.png", temperature=0)
        #     print(pred)
    else:
        output_layer = TokenOutput(vocab_size, vocab_dict)
        output_layer.adapt(train_ds.map(lambda inputs, labels: labels))
        
        model = OCRModel(output_layer, 
                         dictionary=vocab_dict, 
                         embedding_weights=None, 
                         image_shape=image_shape, 
                         patch_shape=patch_shape, 
                         vocab_size=vocab_size,
                         num_heads=8,
                         num_layers=4, 
                         units=512,
                         max_length=32, 
                         dropout_rate=0.2)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=masked_loss,
        metrics=[masked_acc])        

        history = model.fit(
            train_ds.repeat(),
            steps_per_epoch=2000,
            validation_data=valid_ds.repeat(),
            validation_steps=400,
            epochs=100,
            callbacks=callbacks)
        
if __name__ == '__main__':

    train_model(ds_path="dataset/records_short_new", 
                dictionary_path="dataset/vocab.json", 
                patch_shape = [25, 25], 
                image_shape=[150, 450], 
                batch_size=32, 
                weights=None)
