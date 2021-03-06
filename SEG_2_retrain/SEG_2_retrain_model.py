import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import datetime



physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

path = os.path.dirname(os.path.abspath(__file__))

dataset_name = "SEG_2_retrain"



timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
version_dir = path + "/version/" + timestamp 

#os.makedirs(version_dir)
#timestamp



vocabulary = np.genfromtxt("{}/static/vocabulary.csv".format(path), delimiter="\n", dtype=np.int64)
vocab_size = vocabulary.shape[0]
vocab_size



param_list = dict()

param_list["PAST_HISTORY"] = 16
param_list["FUTURE_TARGET"] = 8
param_list["BATCH_SIZE"] = 128
param_list["EPOCHS"] = 10000
param_list["BUFFER_SIZE"] = 200000
param_list["VOCAB_SIZE"] = vocab_size
param_list["LEARNING_RATE"] = 0.01
param_list["NUM_1_NEURONS"] = 177
param_list["NUM_2_NEURONS"] = 177
param_list["DROPOUT_1"] = 0.1
param_list["DROPOUT_2"] = 0.2



train_set = np.genfromtxt("{}/data/{}_train_set.csv".format(path, dataset_name), delimiter="\n", dtype=np.int64)



x_train = tf.data.Dataset.from_tensor_slices(train_set[:-param_list["FUTURE_TARGET"]]).window(param_list["PAST_HISTORY"], 1, 1, True)
# As dataset.window() returns "dataset", not "tensor", need to flat_map() it with sequence length
x_train = x_train.flat_map(lambda x: x.batch(param_list["PAST_HISTORY"])) 
x_train = x_train.map(lambda x: tf.one_hot(x, param_list["VOCAB_SIZE"], axis=-1))
x_train = x_train.batch(param_list["BATCH_SIZE"])



y_train = tf.data.Dataset.from_tensor_slices(train_set[param_list["PAST_HISTORY"]:]).window(param_list["FUTURE_TARGET"], 1, 1, True)
y_train = y_train.flat_map(lambda y: y.batch(param_list["FUTURE_TARGET"]))
y_train = y_train.map(lambda y: tf.one_hot(y, param_list["VOCAB_SIZE"], axis=-1))
y_train = y_train.batch(param_list["BATCH_SIZE"])



train_data = tf.data.Dataset.zip((x_train, y_train))



val_set = np.genfromtxt("{}/data/{}_val_set.csv".format(path, dataset_name), delimiter="\n", dtype=np.int64)



x_val = tf.data.Dataset.from_tensor_slices(val_set[:-param_list["FUTURE_TARGET"]]).window(param_list["PAST_HISTORY"], 1, 1, True)
x_val = x_val.flat_map(lambda x: x.batch(param_list["PAST_HISTORY"]))
x_val = x_val.map(lambda x: tf.one_hot(x, param_list["VOCAB_SIZE"], axis=-1))
x_val = x_val.batch(param_list["BATCH_SIZE"])



y_val = tf.data.Dataset.from_tensor_slices(val_set[param_list["PAST_HISTORY"]:]).window(param_list["FUTURE_TARGET"], 1, 1, True)
y_val = y_val.flat_map(lambda y: y.batch(param_list["FUTURE_TARGET"]))
y_val = y_val.map(lambda y: tf.one_hot(y, param_list["VOCAB_SIZE"], axis=-1))
y_val = y_val.batch(param_list["BATCH_SIZE"])



val_data = tf.data.Dataset.zip((x_val, y_val))



model = keras.models.Sequential()
model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(param_list["NUM_1_NEURONS"])))
model.add(keras.layers.Dropout(param_list["DROPOUT_1"]))
model.add(keras.layers.RepeatVector(param_list["FUTURE_TARGET"]))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(param_list["NUM_2_NEURONS"], return_sequences=True)))
model.add(keras.layers.Dropout(param_list["DROPOUT_2"]))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(param_list["VOCAB_SIZE"], activation='softmax')))

model.compile(optimizer=keras.optimizers.Nadam(param_list["LEARNING_RATE"]), loss='categorical_crossentropy', metrics=['accuracy'])



model_history = model.fit(train_data, batch_size=param_list["BATCH_SIZE"], epochs=param_list["EPOCHS"], validation_data=val_data, callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=35)])


#model.save("{}/version/{}".format(path, timestamp))
model.save(version_dir)
print(len(model_history.history["loss"]))




