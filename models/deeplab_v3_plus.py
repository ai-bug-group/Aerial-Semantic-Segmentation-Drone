from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers import Softmax, Reshape
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras import backend as K

from backbone.mobilenetV2 import mobilenetV2


# 深度可分离空洞卷积 = depthwise + pointwise + atrous conv
def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    # 计算padding的数量，hw是否需要收缩
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    # 如果需要激活函数
    if not depth_activation:
        x = Activation('relu')(x)

    # 分离卷积，首先3x3分离卷积，再1x1卷积
    # 3x3采用膨胀卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    # 1x1卷积，进行压缩
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def Deeplabv3(input_shape=(300, 200, 3), classes=23, alpha=1.):
    img_input = Input(shape=input_shape)

    """
    Encoder: DCNN
    """
    # (25, 38, 320)
    x, skip1 = mobilenetV2(img_input, alpha)
    # 25，38
    size_before = tf.keras.backend.int_shape(x)

    """
    Encoder: 1*1 conv + 3*3 atrous conv(with different rate:6,12,18) + image pooling 
    """
    # 全部求平均后，再利用expand_dims扩充维度，1x1
    # shape = 320
    b4 = GlobalAveragePooling2D()(x)

    # 1x320
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    # 1x1x320
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)

    # 压缩filter
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)

    # 直接利用resize_images扩充hw
    # b4 = 25,38,256
    b4 = Lambda(lambda x: tf.image.resize_images(x, size_before[1:3]))(b4)
    # 调整通道
    # b0-b4的维度都会一样（后面要全部接在一起）
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # SepConv_BN为先3x3膨胀卷积，再1x1卷积，进行压缩
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=6, depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=12, depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=18, depth_activation=True, epsilon=1e-5)

    # 和论文里面一致，全部接在一起再加一个1x1的卷积算压缩是完成了encoder的部分
    # 25,38,256*5=1280
    x = Concatenate()([b4, b0, b1, b2, b3])

    # 利用conv2d压缩
    # 25,38,256
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    """
    low-level feature(其实是原始图片的信息，但是经过了主干网络)
    """
    # skip1.shape[1:3] 为 50,75
    # 50,75,256
    x = Lambda(lambda xx: tf.image.resize_images(x, skip1.shape[1:3]))(x)

    """
    Decoder: 
    """
    # 50, 75, 48
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)

    # 50,57,256+48=304
    x = Concatenate()([x, dec_skip1])
    # 50,75,256
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    # 50,75,256
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    # 300,200,3，取出最终需要还原的图片大小
    size_before3 = tf.keras.backend.int_shape(img_input)

    # 50,75,23
    x = Conv2D(classes, (1, 1), padding='same')(x)
    # 上采样回原始大小+23个语义类别(300,200,23)
    x = Lambda(lambda xx: tf.image.resize_images(xx, size_before3[1:3]))(x)

    # =flatten，(6000, 23)
    x = Reshape((-1, classes))(x)
    # 一张图片上每一个像素点，在23个类别的softmax概率分布，一共6000个点
    x = Softmax()(x)

    inputs = img_input
    model = Model(inputs, x, name='deeplabv3plus')

    return model
