# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:inception.py
# software: PyCharm

import tensorflow.keras.layers as layers
import tensorflow.keras as keras


def inception_block(inputs, filter1, filter3_reduce, filter3, filter5_reduce, filter5, filter_pool, depth=2):
    x1 = layers.Conv2D(filter1, 1, padding='same', activation='relu')(inputs)

    x2 = layers.Conv2D(filter3_reduce, 1, padding='same', activation='relu')(inputs)
    x2 = layers.Conv2D(filter3, 3, padding='same', activation='relu')(x2)

    x3 = layers.Conv2D(filter5_reduce, 1, padding='same', activation='relu')(inputs)
    x3 = layers.Conv2D(filter5, 5, padding='same', activation='relu')(x3)

    x4 = layers.MaxPool2D(3, strides=1, padding='same')(inputs)
    x4 = layers.Conv2D(filter_pool, 1, padding='same', activation='relu')(x4)

    x5 = layers.Concatenate()([x1, x2, x3, x4])

    return x5


def inception_v1(inputs):
    """Inception
    1.discard dense layer that has too many params
    2.dense layer can overfit
    3.network in network: Use conv_1X1 to reduce params and use conv_3X3, conv_5X5 to make
      [receptive field rich]. Make feature rich.

    """
    x = layers.Conv2D(64, 7, 2, padding='same', activation='relu')(inputs)
    x = layers.MaxPool2D(3, 2, padding='same')(x)
    x = layers.Conv2D(64, 1, padding='same', activation='relu')(x)
    x = layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(3, 2, padding='same')(x)

    x1 = inception_block(x, 64, 96, 128, 16, 32, 32)
    x2 = inception_block(x1, 128, 128, 192, 32, 96, 64)
    x3 = layers.MaxPool2D(3, strides=2, padding='same')(x2)
    x4 = inception_block(x3, 192, 96, 208, 16, 48, 64)
    x5 = inception_block(x4, 160, 112, 224, 24, 64, 64)
    x6 = inception_block(x5, 128, 128, 256, 24, 64, 64)
    x7 = inception_block(x6, 112, 144, 288, 32, 64, 64)
    x8 = inception_block(x7, 256, 160, 320, 32, 128, 128)

    x9 = layers.MaxPool2D(3, 2, padding='same')(x8)

    x10 = inception_block(x9, 256, 160, 320, 32, 128, 128)
    x11 = inception_block(x10, 384, 192, 384, 48, 128, 128)

    x12 = layers.AvgPool2D(7, strides=1)(x11)
    x13 = layers.Dropout(rate=0.4)(x12)
    x14 = layers.Dense(1000, activation='softmax')(x13)

    return x14


if __name__ == '__main__':

    img_input = keras.Input(shape=(224, 224, 3))

    outputs = inception_v1(img_input)

    inception = keras.Model(img_input, outputs)

    inception.summary()
