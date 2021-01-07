# -*- coding: utf-8 -*-
# @Time    : 2021/01/05
# @Author  : young
# @File    : deeplab_v3+.py
# @Software: PyCharm

import tensorflow as tf


class DeeplabV3Plus(tf.keras.Model):
    """
    DeeplabV3+ tf2.0+网络架构
    """
    def get_config(self):
        pass

    def __init__(self, num_classes, **kwargs):
        """

        :param num_classes:语义类别
        :param backbone:主干网络（目前支持）
        :param kwargs:
        """
        super(DeeplabV3Plus, self).__init__()

        self.num_classes = num_classes
        self.aspp = None
        self.backbone_feature_1, self.backbone_feature_2 = None, None
        self.input_b_conv, self.conv1, self.conv2, self.out_conv = (None,
                                                                    None,
                                                                    None,
                                                                    None)

    @staticmethod
    def _get_conv_block(filters, kernel_size, conv_activation=None):
        return CustomConv(filters, kernel_size=kernel_size, padding='same',
                          conv_activation=conv_activation,
                          kernel_initializer=tf.keras.initializers.he_normal(),
                          use_bias=False, dilation_rate=1)

    @staticmethod
    def _get_backbone_feature(name_of_layer: str, input_shape) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=input_shape[1:])
        # 这里直接调用官方的代码，减少了bug出现的风险，兼容性也更好
        backbone_model = tf.keras.applications.MobileNetV2(input_tensor=inputs,
                                                           weights='imagenet',
                                                           include_top=False)
        # backbone_model = tf.keras.applications.ResNet50(input_tensor=inputs,
        #                                                 weights='imagenet',
        #                                                 include_top=False)
        outputs = backbone_model.get_layer(name_of_layer).output
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def build(self, input_shape):
        self.backbone_feature_1 = self._get_backbone_feature('out_relu',
                                                             input_shape)
        self.backbone_feature_2 = self._get_backbone_feature('block_3_depthwise_relu',
                                                             input_shape)

        self.aspp = AtrousSpatialPyramidPooling()

        self.input_b_conv = DeeplabV3Plus._get_conv_block(48,
                                                          kernel_size=(1, 1))

        self.conv1 = DeeplabV3Plus._get_conv_block(256,
                                                   kernel_size=3,
                                                   conv_activation='relu')

        self.conv2 = DeeplabV3Plus._get_conv_block(256,
                                                   kernel_size=3,
                                                   conv_activation='relu')

        self.out_conv = tf.keras.layers.Conv2D(self.num_classes,
                                               kernel_size=(1, 1),
                                               padding='same')

    def call(self, inputs, training=None, mask=None):
        image_shape = inputs.shape
        input_a = self.backbone_feature_1(inputs)
        input_b = self.backbone_feature_2(inputs)
        size_before = input_b.shape

        input_a = self.aspp(input_a)
        # 修改
        input_a = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, size_before[1:3]))(input_a)

        input_b = self.input_b_conv(input_b)
        tensor = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        tensor = self.conv2(self.conv1(tensor))

        # 修改
        tensor = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, image_shape[1:3]))(tensor)
        tensor = self.out_conv(tensor)
        tensor = tf.keras.layers.Activation('softmax')(tensor)
        return tensor


class CustomConv(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, padding, dilation_rate,
                 kernel_initializer, use_bias, conv_activation=None):
        """
        定制的2d卷积层（2d_cnv+bn+relu）
        :param n_filters:输出层维度
        :param kernel_size:卷积核大小
        :param padding:padding的
        :param dilation_rate:膨胀率
        :param kernel_initializer:参数初始化的分布
        :param use_bias:是否+b
        :param conv_activation:是否+激活层
        """
        super(CustomConv, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            n_filters, kernel_size=kernel_size, padding=padding,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias, dilation_rate=dilation_rate,
            activation=conv_activation)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        tensor = self.conv(inputs)
        tensor = self.batch_norm(tensor)
        tensor = self.relu(tensor)
        return tensor


class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
    """
    ASPP:空洞卷积金字塔池化
    """
    def __init__(self):
        super(AtrousSpatialPyramidPooling, self).__init__()
        """
        deeplabv3+的模型架构
        """
        self.avg_pool = None
        self.conv1, self.conv2 = None, None
        self.pool = None
        self.out1, self.out6, self.out12, self.out18 = None, None, None, None

    @staticmethod
    def _get_conv_block(kernel_size, dilation_rate, use_bias=False):
        """

        :param kernel_size:
        :param dilation_rate:
        :param use_bias:
        :return:
        """
        return CustomConv(256,
                          kernel_size=kernel_size,
                          dilation_rate=dilation_rate,
                          padding='same',
                          use_bias=use_bias,
                          kernel_initializer=tf.keras.initializers.he_normal())

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        # 原作者是建了个随机同shape的tensor为了知道任意shape的feature map被下采样后的形状
        # 从而利用前后的倍率来告诉tf.keras.layers.UpSampling2D需要怎样还原
        # 但是这样没法还原奇数的shape，这里我已经优化，但是这里我还没更新
        # todo
        dummy_tensor = tf.random.normal(input_shape)

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(input_shape[-3],
                                                                    input_shape[-2]))

        self.conv1 = AtrousSpatialPyramidPooling._get_conv_block(kernel_size=1,
                                                                 dilation_rate=1,
                                                                 use_bias=True)

        self.conv2 = AtrousSpatialPyramidPooling._get_conv_block(kernel_size=1,
                                                                 dilation_rate=1)

        dummy_tensor = self.conv1(self.avg_pool(dummy_tensor))

        self.pool = tf.keras.layers.UpSampling2D(
            size=(
                input_shape[-3] // dummy_tensor.shape[1],
                input_shape[-2] // dummy_tensor.shape[2]
            ),
            interpolation='bilinear'
        )

        self.out1, self.out6, self.out12, self.out18 = map(
            lambda tup: AtrousSpatialPyramidPooling._get_conv_block(kernel_size=tup[0],
                                                                    dilation_rate=tup[1]),
            [(1, 1), (3, 6), (3, 12), (3, 18)]
        )

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        tensor = self.avg_pool(inputs)
        tensor = self.conv1(tensor)
        tensor = tf.keras.layers.Concatenate(axis=-1)([
            self.pool(tensor),
            self.out1(inputs),
            self.out6(inputs),
            self.out12(inputs),
            self.out18(inputs)
        ])
        tensor = self.conv2(tensor)
        return tensor
