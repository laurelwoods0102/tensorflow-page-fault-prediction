# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp_api
import kerastuner
import numpy as np
import pandas as pd
import os
import json
import datetime
import dill


# %%
from tensorflow.keras.layers import (
    Dense, 
    Dropout,
    LSTMCell,
    RNN
)


# %%
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# %%
path = os.path.dirname(os.path.abspath(__file__))

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = path + "/logs/" + timestamp
version_dir = path + "/version/" + timestamp 

os.makedirs(log_dir)
os.makedirs(version_dir)
timestamp


# %%
dataset_name = "SEG_AR_Multiple"


# %%
static_params = {
    'PAST_HISTORY': 16,
    'FUTURE_TARGET': 8,
    'BATCH_SIZE': 512,
    'BUFFER_SIZE': 200000,
    'EPOCHS': 500,
    'VOCAB_SIZE': 16293
 }


# %%
hparams_simple = {
    "HP_LSTM_1_UNITS" : 128,
    "HP_LSTM_1_DROPOUT" : 0.0,
    "HP_LEARNING_RATE" : 1e-3,
}


# %%
hparams_multiple = {
    "HP_LSTM_1_UNITS" : 32,
    "HP_LSTM_2_UNITS" : 32,
    "HP_LSTM_1_DROPOUT" : 0.0,
    "HP_LSTM_2_DROPOUT" : 0.0,
    "HP_LEARNING_RATE" : 1e-3,
}


# %%
def generate_timeseries(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, n_feature)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        #data.append(dataset[indices])
        labels.append(np.reshape(dataset[i:i+target_size], (target_size, 1)))
        #labels.append(dataset[i:i+target_size])
    return np.array(data), np.array(labels)


# %%
train_set = np.genfromtxt(path + "/data/SEG_train_set.csv", delimiter="\n", dtype=np.int32)
x_train, y_train = generate_timeseries(train_set, 0, None, static_params["PAST_HISTORY"], static_params["FUTURE_TARGET"])
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().batch(static_params["BATCH_SIZE"]).shuffle(static_params["BUFFER_SIZE"])


# %%
val_set = np.genfromtxt(path + "/data/SEG_val_set.csv", delimiter="\n", dtype=np.int32)
x_val, y_val = generate_timeseries(val_set, 0, None, static_params["PAST_HISTORY"], static_params["FUTURE_TARGET"])
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.cache().batch(static_params["BATCH_SIZE"])


# %%
class SEGARSimple(keras.Model):
    def __init__(self, units, dropout, output_steps, output_size):
        super().__init__()
        self.output_steps = output_steps
        self.units = units
        self.lstm_cell = LSTMCell(units, dropout=dropout)

        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        self.dense = Dense(output_size, activation="softmax")

    @tf.function
    def warmup(self, inputs):
        onehot_inputs = tf.squeeze(tf.one_hot(inputs, static_params["VOCAB_SIZE"]), axis=2)

        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(onehot_inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)

        return prediction, state

    @tf.function
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        #predictions = []
        predictions = tf.TensorArray(tf.float32, size=self.output_steps, clear_after_read=False)
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        #predictions.append(prediction)
        predictions = predictions.write(0, prediction)

        # Run the rest of the prediction steps
        for i in tf.range(1, self.output_steps):
            # Use the last prediction as input.
            x = prediction

            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)

            # Convert the lstm output to a prediction.
            prediction = self.dense(x)

            # Add the prediction to the output
            #predictions.append(prediction)
            predictions = predictions.write(i, prediction)

        # predictions.shape => (time, batch, features)
        #predictions = tf.stack(predictions)
        predictions = predictions.stack()

        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


# %%
class SEGARMultiple(keras.Model):
    def __init__(self, units_1, units_2, dropout_1, dropout_2, output_steps, output_size):
        super().__init__()
        self.output_steps = output_steps
        self.units_1 = units_1
        self.units_2 = units_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

        self.lstm_cell_1 = LSTMCell(units_1, dropout=dropout_1)
        self.lstm_cell_2 = LSTMCell(units_2, dropout=dropout_2)

        self.lstm_rnn_1 = RNN(self.lstm_cell_1, return_state=True, return_sequences=True)
        self.lstm_rnn_2 = RNN(self.lstm_cell_2, return_state=True)
        self.dense = Dense(output_size, activation="softmax")

    @tf.function#(input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.int32)])
    def warmup(self, inputs):
        onehot_inputs = tf.squeeze(tf.one_hot(inputs, static_params["VOCAB_SIZE"]), axis=2)

        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x_1, *state_1 = self.lstm_rnn_1(onehot_inputs)
        x_2, *state_2 = self.lstm_rnn_2(x_1)

        # predictions.shape => (batch, features)
        prediction = self.dense(x_2)

        return prediction, state_1, state_2

    @tf.function#(input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.int32)])
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        #predictions = []
        predictions = tf.TensorArray(tf.float32, size=self.output_steps, clear_after_read=False)

        # Initialize the lstm state
        prediction, state_1, state_2 = self.warmup(inputs)

        # Insert the first prediction
        #predictions.append(prediction)
        predictions = predictions.write(0, prediction)

        # Run the rest of the prediction steps
        for i in tf.range(1, self.output_steps):
            # Use the last prediction as input.
            x = prediction

            # Execute one lstm step.
            x_1, state_1 = self.lstm_cell_1(x, states=state_1, training=training)
            x_2, state_2 = self.lstm_cell_2(x_1, states=state_2, training=training)

            # Convert the lstm output to a prediction.
            prediction = self.dense(x_2)

            # Add the prediction to the output
            #predictions.append(prediction)
            predictions = predictions.write(i, prediction)

        # predictions.shape => (time, batch, features)
        #predictions = tf.stack(predictions)
        predictions = predictions.stack()

        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


# %%
model = SEGARSimple(
    units=hparams_multiple["HP_LSTM_1_UNITS"], dropout=hparams_multiple["HP_LSTM_1_DROPOUT"], 
    output_steps=static_params["FUTURE_TARGET"], output_size=static_params["VOCAB_SIZE"])

# %% [markdown]
# model = SEGARMultiple(
#     units_1=hparams_multiple["HP_LSTM_1_UNITS"], units_2=hparams_multiple["HP_LSTM_2_UNITS"], dropout_1=hparams_multiple["HP_LSTM_1_DROPOUT"], 
#     dropout_2=hparams_multiple["HP_LSTM_2_DROPOUT"], output_steps=static_params["FUTURE_TARGET"], output_size=static_params["VOCAB_SIZE"])

# %%
model.compile(
    optimizer=keras.optimizers.Nadam(hparams_simple["HP_LEARNING_RATE"]),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)


# %%
with open(path + "/static/test_pipeline.pkl", "rb") as p:
    test_pipeline = dill.load(p)

test_set = np.genfromtxt(path + "/data/SEG_test_set_original.csv", delimiter="\n", dtype=np.int64)
processed_test_set = test_pipeline.transform(test_set.copy())
x_test, y_test = generate_timeseries(processed_test_set, 0, None, static_params["PAST_HISTORY"], static_params["FUTURE_TARGET"])

with tf.summary.create_file_writer(log_dir).as_default():
    hp_api.hparams(hparams_simple, trial_id=timestamp)
    history = model.fit(train_data, validation_data=val_data, epochs=1, callbacks=[
        keras.callbacks.EarlyStopping('val_accuracy', patience=5),
        keras.callbacks.TensorBoard(log_dir)
        ])

    loss, acc = model.evaluate(x_test, y_test)
    tf.summary.scalar('test_loss', loss, step=x_test.shape[0])
    tf.summary.scalar('test_accuracy', acc, step=x_test.shape[0])


# %%
model.summary()

tf.saved_model.save(model, version_dir, 
    signatures=model.call.get_concrete_function(tf.TensorSpec(shape=[None, None, 1], dtype=tf.int32, name="call")))

# %%
with open("{}/version/{}/evaluate.csv".format(path, timestamp), "w") as r:
    r.write("loss, accuracy\n")
    r.write("{}, {}".format(loss, acc))