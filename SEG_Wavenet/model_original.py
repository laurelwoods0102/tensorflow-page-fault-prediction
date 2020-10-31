import tensorflow as tf
from tensorflow import keras
import numpy as np

from modules import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic

class SEGWaveNet(tf.keras.Model):
    def __init__(self, batch_size, dilations, filter_width, residual_channels, dilation_channels, skip_channels, out_channels, quantization_channels=2**8, use_biases=False, initial_filter_width=32):

        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.initial_filter_width = initial_filter_width

        self.receptive_field = SEGWaveNet.calculate_receptive_field(self.filter_width, self.dilations, self.initial_filter_width)

    @staticmethod
    def calculate_receptive_field(filter_width, dilations, initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1

        # scalar_input
        receptive_field += initial_filter_width - 1
        return receptive_field

    def _create_causal_layer(self, input_batch):
        with tf.name_scope("causal_layer"):
            return keras.layers.Conv1D(
                input_batch,
                filters=self.residual_channels,
                kernel_size=self.initial_filter_width,
                padding='valid',
                dilation_rate=1,
                use_biase=False
            )

    def _create_queue(self):
        pass

    def _create_dilation_layer(self, input_batch, layer_index, dilation, output_width):
        with tf.name_scope("dilation_layer_{}".format(layer_index)):       #!
            conv_filter = keras.layers.Conv1D(
                input_batch,
                filters=self.dilation_channels,
                kernel_size=self.filter_width,
                dilation_rate=dilation,
                padding='valid',
                use_bias=self.use_biases,
                name="conv_filter"
            )
            conv_gate = keras.layers.Conv1D(
                input_batch,
                filters=self.dilation_channels,
                kernel_size=self.filter_width,
                dilation_rate=dilation,
                padding='valid',
                use_bias=self.use_biases,
                name="conv_gate"
            )

            out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
            
            ## skip_contribution : Summed up to create output
            skip_cut = tf.shape(out)[1] - output_width
            out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, self.dilation_channels])
            skip_contribution = keras.layers.Conv1D(
                out_skip,
                filters=self.skip_channels,
                kernel_size=1,
                padding="same",
                use_bias=self.use_biases,
                name="skip"
            )

            ## transformed : 1x1 conv to out (= gate * filter) to produce residuals (= dense output)
            transformed = keras.layers.Conv1D(
                out,
                filters=self.residual_channels,
                kernel_size=1,
                padding="same",
                use_bias=self.use_biases,
                name="dense"
            )

            input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
            input_batch_cut = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])
            dense_output = input_batch_cut + transformed

            return skip_contribution, dense_output

    def _create_network(self, input_batch):
        if self.train_mode == False:
            self._create_queue()

        outputs = []
        current_layer = input_batch     # Length is reduced by 1 due to causal cut

        if self.train_mode == False:
            self.causal_queue = tf.tensor_scatter_nd_update(
                self.causal_queue, 
                tf.range(self.batch_size),
                tf.concat([self.causal_queue[:, 1:, :], input_batch], axis=1)
                )
            current_layer = self.causal_queue

        current_layer = self._create_causal_layer(current_layer)

        if self.train_mode == True:
            output_width = tf.shape(input_batch)[1] - self.receptive_field + 1
        else:
            output_width = 1

        with tf.name_scope("dilated_stack"):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope("layer_{}".format(layer_index)):
                    if self.train_mode == False:
                        self.dilation_queue[layer_index] = tf.tensor_scatter_nd_update(
                            self.dilation_queue[layer_index],
                            tf.range(self.batch_size),
                            tf.concat([self.dilation_queue[layer_index][:, 1:, :], current_layer], axis=1)
                            )
                        current_layer = self.dilation_queue[layer_index]

                        output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation, output_width)
                        outputs.append(output)

                with tf.name_scope("postprocessing"):
                    total = sum(outputs)
                    transformed_1 = tf.nn.relu(total)
                    conv_1 = keras.layers.Conv1D(
                        transformed_1,
                        filters=self.skip_channels,
                        kernel_size=1,
                        padding="same",
                        use_bias=self.use_biases
                    )

                    transformed_2 = tf.nn.relu(conv_1)
                    conv_2 = keras.layers.Conv1D(
                        transformed_2,
                        filters=self.out_channels,
                        kernel_size=1,
                        padding="same",
                        use_bias=self.use_biases
                    )

                return conv_2

    def _one_hot(self, input_batch):
        with tf.name_scope("one_hot_encode"):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32
                )
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def predict_proba_incremental(self, x, name="wavenet"):
        with tf.name_scope(name):
            encoded = tf.reshape(x, [self.batch_size, -1, 1])

            raw_output = self._create_network(encoded)

            out = tf.reshape(raw_output, [self.batch_size, -1, self.out_channels])
            proba = sample_from_discretized_mix_logistic(out)

            return proba

    def add_loss(self, input_batch, l2_regularization_strength=None, name="wavenet"):
        with tf.name_scope(name):
            encoded = self._one_hot(input_batch)
            encoded = tf.cast(encoded, tf.float32)

            network_input = tf.reshape(encoded, [self.batch_size, -1, 1])

            network_input_width = tf.shape(network_input)[1] - 1

            input = tf.slice(network_input, [0, 0, 0], [-1, network_input_width, 1])

            raw_output = self._create_network(input)

            with tf.name_scope("loss"):
                target_output = tf.slice(network_input, [0, self.receptive_field, 0], [-1, -1, -1])

                loss = discretized_mix_logistic_loss(raw_output, target_output, num_class=2**16, reduce=False)      # num_class : 16 bits or 64bits ?
                reduced_loss = tf.math.reduce_mean(loss)

                tf.summary.scalar('loss', reduced_loss)

                if l2_regularization_strength == None:
                    self.loss = reduced_loss
                else:
                    l2_loss = tf.math.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if not ('bias' in v.name)])   #

                    total_loss = (reduced_loss + l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    self.loss = total_loss

    def add_optimizer(self, hparams, global_steps):
        with tf.name_scope("optimizer"):
            hp = hparams

            learning_rate = keras.optimizers.schedules.ExponentialDecay(hp.learning_rate, global_steps, hp.decay_steps, hp.decay_rate)

            self.learning_rate = learning_rate
            optimizer = keras.optimizers.Nadam(learning_rate)

            with tf.GradientTape() as tape:             #
                loss = self.loss
            variables = self.trainable_variables
            gradients = tape.gradient(loss, variables)

            if hp.clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)
            else:
                clipped_gradients = gradients
