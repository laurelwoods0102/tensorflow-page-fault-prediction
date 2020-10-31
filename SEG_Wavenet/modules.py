import tensorflow as tf
from tensorflow import keras


class ResidualBlock(keras.Model):
    def __init__(self, dilation, filter_width, dilation_channels, residual_channels, skip_channels, use_biases, output_width):
        super().__init__()

        self.dilation = dilation
        self.filter_width = filter_width
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.use_biases = use_biases
        self.output_width = output_width 

        self.conv_filter = keras.layers.Conv1D(
            filters=self.dilation_channels,
            kernel_size=self.filter_width,
            dilation_rate=self.dilation,
            padding='valid',
            use_bias=self.use_biases,
            name="conv_filter"
        )
        self.conv_gate = keras.layers.Conv1D(
            filters=self.dilation_channels,
            kernel_size=self.filter_width,
            dilation_rate=self.dilation,
            padding='valid',
            use_bias=self.use_biases,
            name="conv_gate"
        )
        self.conv_skip = keras.layers.Conv1D(
            filters=self.skip_channels,
            kernel_size=1,
            padding="same",
            use_bias=self.use_biases,
            name="skip"
        )
        ## transformed : 1x1 conv to out (= gate * filter) to produce residuals (= dense output)
        ## conv_residual (=skip_contribution in original)
        self.conv_residual = keras.layers.Conv1D(
            filters=self.residual_channels,
            kernel_size=1,
            padding="same",
            use_bias=self.use_biases,
            name="dense"
        )

    @tf.function
    def call(self, x, training=False):
        out = tf.tanh(self.conv_filter(x)) * tf.sigmoid(self.conv_gate(x))
        
        ## skip_output (=skip contribution in original) : Summed up to create output
        skip_cut = tf.shape(out)[1] - self.output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, self.dilation_channels])
        skip_output = self.conv_residual(out_skip)

        transformed = self.conv_residual(out)
        input_cut = tf.shape(x)[1] - tf.shape(transformed)[1]
        x_cut = tf.slice(x, [0, input_cut, 0], [-1, -1, -1])
        dense_output = x_cut + transformed

        return skip_output, dense_output


