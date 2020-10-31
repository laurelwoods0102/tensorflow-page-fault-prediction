import tensorflow as tf
from tensorflow import keras
import numpy as np

from mixtures import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic



class SEGWaveNet(keras.Model):
    def __init__(self, batch_size, dilations, filter_width, residual_channels, dilation_channels, skip_channels, out_channels, quantization_channels=2**8, use_biases=False, initial_filter_width=32):
        super().__init__()

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

    @tf.function
    def call(self, x, training=False):
        pass
