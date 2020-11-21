import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import custom_layers as clayers

# Necessary parameters are inputs and name

def conv_squeeze(kernel_size, inputs, name=None):
    x = layers.Conv2D(
        1,
        kernel_size,
        padding='same',
        name='final_conv_'+name,
        dtype='float32',
    )(inputs)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32',name=name)(x)
    return outputs

def conv3_16(inputs, name=None):
    x = layers.Conv2D(
        16,
        3,
        padding='same',
        name='final_conv1_'+name,
    )(inputs)
    x = layers.Conv2D(
        16,
        3,
        padding='same',
        name='final_conv2_'+name,
    )(x)
    x = layers.Conv2D(
        1,
        3,
        padding='same',
        name='final_conv_squeeze_'+name,
        dtype='float32'
    )(x)
    x = tf.squeeze(x, axis=-1)
    outputs = layers.Activation('linear', dtype='float32',name=name)(x)
    return outputs 