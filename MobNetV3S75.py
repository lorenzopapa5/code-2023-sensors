import sys

import keras.activations
from tensorflow.keras import Model, models, applications
from loss import accurate_obj_boundaries_loss
from utils import rmse
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.python.keras import backend
import tensorflow as tf


def upsample_layer(tensor, filters, name, concat_with, base_model):
    up_i = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', dilation_rate=1,
                           name=name + '_upconv', use_bias=False)(tensor)
    up_i = Concatenate(name=name + '_concat')([up_i, base_model.get_layer(concat_with).output])  # Skip connection
    up_i = ReLU(name=name + '_relu1')(up_i)
    up_i = layers.SeparableConv2D(filters=filters,
                                  kernel_size=3,
                                  padding='same',
                                  use_bias=False,
                                  name=name + '_sep_conv_1')(up_i)
    up_i = ReLU(name=name + '_relu2')(up_i)

    return up_i

def create_model(input_shape, existing='', initialize=False, freeze=False):
    if len(existing) == 0 and not initialize:
        encoder = tf.keras.applications.MobileNetV3Small(input_shape=input_shape,
                                                         minimalistic=False,
                                                         alpha=0.75,
                                                         include_top=False)
        # encoder.summary()
        print('Number of layers in the encoder: {}'.format(len(encoder.layers)))

        # Starting point for decoder
        base_model_output_shape = encoder.layers[-1].output.shape
        decode_filters = 256

        # Decoder Layers
        decoder_0 = Conv2D(filters=decode_filters,
                           kernel_size=1,
                           padding='same',
                           input_shape=base_model_output_shape,
                           name='conv_Init_decoder')(encoder.layers[-1].output)
        decoder_1 = upsample_layer(decoder_0, int(decode_filters / 2), 'up1', concat_with='expanded_conv_3/depthwise',
                                   base_model=encoder)
        decoder_2 = upsample_layer(decoder_1, int(decode_filters / 4), 'up2', concat_with='expanded_conv_1/depthwise',
                                   base_model=encoder)
        decoder_3 = upsample_layer(decoder_2, int(decode_filters / 8), 'up3', concat_with='expanded_conv/depthwise',
                                   base_model=encoder)
        decoder_4 = upsample_layer(decoder_3, int(decode_filters / 16), 'up4', concat_with='Conv',
                                   base_model=encoder)
        convDepthF = Conv2D(filters=1,
                            kernel_size=3,
                            padding='same',
                            name='convDepthF')(decoder_4)

        # Create the model
        model = Model(inputs=encoder.input, outputs=convDepthF)
        print('Number of layers: {}'.format(len(model.layers)))
        model.summary()

    return model
