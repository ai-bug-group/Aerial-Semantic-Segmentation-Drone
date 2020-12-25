# -*- coding: utf-8 -*-
# @Time    : 2020/11/25
# @Author  : young
# @File    : train.py
# @Software: PyCharm
import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from cv2 import imread, resize
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from base.keras_base import KerasBase
from models.deeplab_v3_plus import Deeplabv3
# from models.FCN_8 import fcn_8
from utils.metric import mean_iou
from utils.path_utils import WEIGHTS_PATH_MOBILE_V2

MULTI_GPU = False
CLASSES = 23
HEIGHT = 640
WIDTH = 320

if MULTI_GPU:
    CUDA_VISIBLE_DEVICES = "2,3,5,6"
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES


def random_crop(image, mask, H=512, W=512):
    """

    :param image:
    :param mask:
    :param H:
    :param W:
    :return:
    """
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - H, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - W, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=H,
                                          target_width=W)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=H,
                                         target_width=W)
    return image, mask


def data_generator(files, batch_size):
    """
    语义分割训练数据生成器
    :param files: 图片数据
    :param batch_size: 批次大小
    :return:
    """
    # 获取总长度
    n = len(files)
    i = 0
    while True:
        x_train = []
        y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i == 0:
                # 刚开始先打乱数据
                np.random.shuffle(files)
            image_path = files[i][0]
            mask_path = files[i][1]
            # 从文件中读取原始图像
            image = imread(image_path)

            # 从文件中读取语义分割的标签
            mask = imread(mask_path)

            # resize
            image = resize(image, (int(HEIGHT), int(WIDTH)))
            mask = resize(mask, (int(HEIGHT), int(WIDTH)))

            # image, mask = random_crop(image, mask, HEIGHT, WIDTH)

            # 生成标签
            seg_labels = np.zeros((int(WIDTH), int(HEIGHT), CLASSES))
            for c in range(CLASSES):
                seg_labels[:, :, c] = (mask[:, :, 0] == c).astype(int)
            seg_labels = np.reshape(seg_labels, (-1, CLASSES))

            x_train.append(image)
            y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i + 1) % n
        yield np.array(x_train), np.array(y_train)


def loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


class SemanticSegmentation(KerasBase):
    def __init__(self):
        super().__init__()
        self.lr_ph = 1e-3
        self.loss = loss
        self.data_generator = data_generator

        # 保存的方式，3个epoch保存一次
        self.checkpoint_period = ModelCheckpoint(
            log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            period=3
        )
        # 学习率下降的方式，val_loss 2次不下降就下降学习率继续训练
        self.reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
        # 是否需要提前停止训练，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=6,
            verbose=1
        )

        self.build_model()

    def build_model(self):
        """
        构建模型的结构
        :return: model: keras的模型结构的封装
                 -type: Model
        """
        self.model = Deeplabv3(classes=CLASSES, input_shape=(WIDTH, HEIGHT, 3))
        if MULTI_GPU:
            self.model = multi_gpu_model(self.model, gpus=len(CUDA_VISIBLE_DEVICES.split(',')))

        weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH_MOBILE_V2,
                                cache_subdir='models')

        self.model.load_weights(weights_path,
                                by_name=True,
                                skip_mismatch=True)

    def train(self,
              data: list,
              batch_size: int,
              lr: float,
              epochs: int,
              initial_epoch: int,
              split_train: int):
        """
        训练
        :param data: 训练数据
               -type: list
               -element: tuple
        :param batch_size: 批次大小
        :param lr: 学习率
        :param epochs: 训练轮次
        :param initial_epoch: 初始轮次
        :param split_train: 训练数据、验证数据切割线
        :return: 暂无
        """
        self.model.compile(loss=loss,
                           optimizer=Adam(lr=lr),
                           metrics=['accuracy', mean_iou])
        self.model.fit_generator(self.data_generator(data[:split_train], batch_size),
                                 steps_per_epoch=max(1, num_train // batch_size),
                                 validation_data=self.data_generator(data[split_train:], batch_size),
                                 validation_steps=max(1, num_val // batch_size),
                                 epochs=epochs,
                                 initial_epoch=initial_epoch,
                                 callbacks=[self.checkpoint_period,
                                            self.reduce_lr,
                                            self.early_stopping])

    def predict(self, data):
        """
        简单的预测测试（待优化）
        :param data:
        :return:
        """
        image = imread(data)
        image = resize(image, (int(HEIGHT), int(WIDTH)))
        image = np.expand_dims(image, axis=0)
        logits = self.model.predict(image, steps=1)
        logits = np.squeeze(logits)
        masks = np.argmax(logits, axis=1)
        masks = np.reshape(masks, (int(WIDTH), int(HEIGHT)))
        cv2.imwrite("data/test_mask_result.png", masks)
        return masks


if __name__ == "__main__":
    is_training = True

    data_dir = "data/all/"
    log_dir = "checkpoints/"

    # 读取数据
    image_list = sorted(glob(data_dir + 'images/*'))
    mask_list = sorted(glob(data_dir + 'masks/*'))
    aerial_data = list(zip(image_list, mask_list))

    # 90%用于训练，10%用于验证。
    num_val = int(len(aerial_data) * 0.1)
    num_train = len(aerial_data) - num_val

    # 搭建语义分割模型
    ss = SemanticSegmentation()
    ss.print_model_summary()

    if is_training:
        # 训练策略
        bz1 = 4
        lr1 = 1e-3
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, bz1))
        ss.train(
            aerial_data,
            bz1,
            lr1,
            30,
            0,
            num_train
        )
        ss.save_weights_only(log_dir + 'middle1.h5')

        bz2 = 8
        lr2 = 1e-4
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, bz2))
        ss.train(
            aerial_data,
            bz2,
            lr2,
            60,
            30,
            num_train
        )
        ss.save_weights_only(log_dir + 'last1.h5')
    else:
        test_image_dir = 'data/all/images/000.jpg'
        checkpoint = 'checkpoints/test_model.h5'
        ss.model.load_weights(checkpoint)
        predicts = ss.predict(test_image_dir)
        print(np.min(predicts))
