import tensorflow as tf
from text_recognition.ocr_model import OCRModel
from text_recognition.embedding_utils import load_json_data
from text_recognition.custom_layers import TokenOutput
from text_recognition.dataset import prepare_dataset

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


def train(train_ds_path, valid_ds_path, vocab_size=409094, dictionary_path="dataset/vocab_bpemb.json"):
  
  train_ds = tf.data.TFRecordDataset(train_ds_path)
  valid_ds = tf.data.TFRecordDataset(valid_ds_path)

  train_ds = prepare_dataset(train_ds)
  valid_ds = prepare_dataset(valid_ds)

  callbacks = [
    GenerateText(),
    tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)]
  
  vocab_dict = load_json_data(dictionary_path)
  output_layer = TokenOutput(vocab_size, vocab_dict)
  
  model = OCRModel(output_layer, dictionary=vocab_dict, vocab_seize=vocab_size)

  history = model.fit(
    train_ds.repeat(),
    steps_per_epoch=100,
    validation_data=valid_ds.repeat(),
    validation_steps=20,
    epochs=100,
    callbacks=callbacks)
  
