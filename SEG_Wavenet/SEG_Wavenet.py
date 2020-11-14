import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)



from mixtures import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic



class DiscretizedMixLogisticLoss(keras.losses.Loss):
    def __init__(self, name="discretized_mix_logistic_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss =  discretized_mix_logistic_loss(y_pred, y_true)
        return tf.reduce_mean(loss)



class Conv1D(keras.layers.Conv1D):
    def __init__(self, filters, kernel_size, strides=1, padding="causal", dilation_rate=1, use_bias=False, *args, **kwargs):
        super().__init__(filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)
        
        ## (issue) Set name other than k and d invoke error : TypeError: unsupported operand type(s) for +: 'int' and 'tuple'
        self.k = kernel_size                
        self.d = dilation_rate

        self.use_bias = use_bias

        if kernel_size > 1:
            self.current_receptive_field = kernel_size + (kernel_size - 1) * (dilation_rate - 1)       # == queue_len (tf2)
            self.residual_channels = residual_channels
            self.queue = tf.zeros([1, self.current_receptive_field, filters])

    def build(self, x_shape):
        super().build(x_shape)

        self.linearized_weights = tf.cast(tf.reshape(self.kernel, [-1, self.filters]), dtype=tf.float32)

    def call(self, x, training=False):
        if not training:
            return super().call(x)

        if self.kernel_size > 1:
            self.queue = self.queue[:, 1:, :]
            self.queue = tf.concat([self.queue, tf.expand_dims(x[:, -1, :], axis=1)], axis=1)

            if self.dilation_rate > 1:
                x = self.queue[:, 0::self.d, :]
            else:
                x = self.queue

            outputs = tf.matmul(tf.reshape(x, [1, -1]), self.linearized_weights)
            
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias)

            return tf.reshape(outputs, [-1, 1, self.filters])

    #def init_queue(self):
        



class ResidualBlock(keras.Model):
    def __init__(self, layer_index, dilation, filter_width, dilation_channels, residual_channels, skip_channels, use_biases, output_width):
        super().__init__()

        self.layer_index = layer_index
        self.dilation = dilation
        self.filter_width = filter_width
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.use_biases = use_biases
        self.output_width = output_width

    def build(self, input_shape):
        self.conv_filter = keras.layers.Conv1D(
            filters=self.dilation_channels,
            kernel_size=self.filter_width,
            dilation_rate=self.dilation,
            padding='valid',
            use_bias=self.use_biases,
            name="residual_block_{}/conv_filter".format(self.layer_index)
        )
        self.conv_gate = keras.layers.Conv1D(
            filters=self.dilation_channels,
            kernel_size=self.filter_width,
            dilation_rate=self.dilation,
            padding='valid',
            use_bias=self.use_biases,
            name="residual_block_{}/conv_gate".format(self.layer_index)
        )
        ## transformed : 1x1 conv to out (= gate * filter) to produce residuals (= dense output)
        ## conv_residual (=skip_contribution in original)
        self.conv_residual = keras.layers.Conv1D(
            filters=self.residual_channels,
            kernel_size=1,
            padding="same",
            use_bias=self.use_biases,
            name="residual_block_{}/dense".format(self.layer_index)
        )
        self.conv_skip = keras.layers.Conv1D(
            filters=self.skip_channels,
            kernel_size=1,
            padding="same",
            use_bias=self.use_biases,
            name="residual_block_{}/skip".format(self.layer_index)
        )


    @tf.function
    def call(self, inputs, training=False):
        out = tf.tanh(self.conv_filter(inputs)) * tf.sigmoid(self.conv_gate(inputs))
        
        if training:
            skip_cut = tf.shape(out)[1] - self.output_width
        else:
            skip_cut = tf.shape(out)[1] - 1
            
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, self.dilation_channels])
        skip_output = self.conv_skip(out_skip)

        transformed = self.conv_residual(out)
        input_cut = tf.shape(inputs)[1] - tf.shape(transformed)[1]
        x_cut = tf.slice(inputs, [0, input_cut, 0], [-1, -1, -1])
        dense_output = x_cut + transformed

        return skip_output, dense_output



class PostProcessing(keras.Model):
    def __init__(self, skip_channels, quantization_channels, use_biases):
        super().__init__()

        self.skip_channels = skip_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases

    def build(self, input_shape):
        self.conv_1 = keras.layers.Conv1D(
            filters=self.skip_channels,
            kernel_size=1,
            padding="same",
            use_bias=self.use_biases,
            name="postprocessing/conv_1"
        )
        self.conv_2 = keras.layers.Conv1D(
            #filters=self.out_channels,          # For Discretized MoL Parameterization
            filters=self.quantization_channels,
            kernel_size=1,
            padding="same",
            use_bias=self.use_biases,
            name="postprocessing/conv_2"
        )
    
    @tf.function
    def call(self, inputs, training=False):
        x = tf.nn.relu(inputs)
        x = self.conv_1(x)

        x = tf.nn.relu(x)
        x = self.conv_2(x)

        return x



class WaveNet(keras.Model):
    def __init__(self, batch_size, dilations, filter_width, dilation_channels, residual_channels, skip_channels, quantization_channels=None, out_channels=None, use_biases=False):
        super().__init__()

        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        #self.initial_filter_width = initial_filter_width
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.quantization_channels = quantization_channels
        self.out_channels = out_channels
        self.use_biases = use_biases

        # Scalar Input receptive field
        #self.receptive_field = (self.filter_width - 1) * sum(self.dilations) + self.initial_filter_width

        # Onehot Input Receptive Field
        self.receptive_field = (self.filter_width - 1) * sum(self.dilations) + self.filter_width

    def build(self, input_shape):   
        self.output_width = input_shape[1] - self.receptive_field + 1       # total output width of model

        self.preprocessing_layer = keras.layers.Conv1D(
            filters=self.residual_channels,
            #kernel_size=self.initial_filter_width,
            kernel_size=self.filter_width,
            use_bias=self.use_biases,
            name="preprocessing/conv")

        self.residual_blocks = []
        for _ in range(1):
            for i, dilation in enumerate(self.dilations):
                self.residual_blocks.append(
                    ResidualBlock(
                        layer_index=i,
                        dilation=self.dilations[0], 
                        filter_width=self.filter_width, 
                        dilation_channels=self.dilation_channels, 
                        residual_channels=self.residual_channels, 
                        skip_channels=self.skip_channels, 
                        use_biases=self.use_biases, 
                        output_width=self.output_width)
                    )

        self.postprocessing_layer = PostProcessing(self.skip_channels, self.quantization_channels, self.use_biases)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        #inputs = tf.sparse.to_dense(inputs)     # x from onehot dataset
        inputs = tf.one_hot(inputs, self.quantization_channels, axis=-1)

        x = self.preprocessing_layer(inputs)
        skip_outputs = []

        for layer_index in range(len(self.dilations)):
            skip_output, x = self.residual_blocks[layer_index](x, training=training)
            skip_outputs.append(skip_output)
      
        skip_sum = tf.math.add_n(skip_outputs)
        
        output = self.postprocessing_layer(skip_sum)
        
        if not training:
            out = tf.reshape(output, [self.batch_size, -1, self.quantization_channels])
            #output = sample_from_discretized_mix_logistic(out)
            output = tf.cast(tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)

        return output

    def train_step(self, data): 
        x, y = data
        y = tf.one_hot(y, self.quantization_channels, axis=-1)
        
        #y = tf.expand_dims(tf.sparse.to_dense(y), axis=1)      # y from onehot dataset

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            reduced_loss = tf.math.reduce_mean(loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    #@tf.function(experimental_relax_shapes=True)
    def test_step(self, data):
        x, y = data
        y = tf.one_hot(y, self.quantization_channels, axis=-1)

        y_pred = self(x, training=False)

        loss = self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}



dataset_name = "SEG_wavenet"
version = 2


# HParms follows the Diagram 
batch_size = 1
dilations = [1, 2, 4, 8, 16, 32, 64, 128]
filter_width = 2        # == kernel_size
#initial_filter_width = 32       # from (tacokr)
dilation_channels = 32  # unknown
residual_channels = 24
skip_channels = 128
#quantization_channels = 2**8
quantization_channels = 16293   # == vocab_size
#out_channels = 10*3
use_biases = False

#receptive_field = 287      # Scalar Input
receptive_field = 257       # Onehot Input


#wavenet.receptive_field = 287
#wavenet.output_width = 738



train_set_original = np.genfromtxt("{}/data/{}_train_set.csv".format(path, dataset_name), delimiter="\n", dtype=np.int64)



train_y_set_original = train_set_original.copy()
train_y_set_original = train_y_set_original[receptive_field:]



train_y = train_y_set_original.reshape(-1, 1)



train_y.shape



train_x_set_original = train_set_original.copy()
train_x_set_original = train_x_set_original[:-1]



train_x = []

for i in range(len(train_x_set_original) - receptive_field + 1):
    train_x.append(train_x_set_original[i:i+receptive_field])



train_x = np.array(train_x)



train_x.shape



wavenet = WaveNet(batch_size, dilations, filter_width, dilation_channels, residual_channels, skip_channels, quantization_channels)



wavenet.compile(keras.optimizers.Nadam(), loss=keras.losses.CategoricalCrossentropy(from_logits=True))



wavenet.fit(train_x, train_y, batch_size=128, epochs=300, callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=5)])



wavenet.save("{}/model/model_{}".format(path, version))



test_set_original = np.genfromtxt("{}/data/{}_test_set.csv".format(path, dataset_name), delimiter="\n", dtype=np.int64)



test_y_set_original = test_set_original.copy()
test_y_set_original = test_y_set_original[receptive_field:]



test_y = test_y_set_original.reshape(-1, 1)



#test_y = np.eye(quantization_channels)[test_y]
#test_y = np.squeeze(test_y, axis=1)



test_y.shape



test_x_set_original = test_set_original.copy()
test_x_set_original = test_x_set_original[:-1]



test_x = []

for i in range(len(test_x_set_original) - receptive_field + 1):
    test_x.append(test_x_set_original[i:i+receptive_field])



test_x = np.array(test_x)



test_x.shape



result = wavenet.evaluate(test_x, test_y, batch_size=128)



with open("{}/model/model_{}_result.csv".format(path, version), "w") as r:
    r.write(str(result))

