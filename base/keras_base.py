# -*- coding: utf-8 -*-
# @Time    : 2019/11/1 10:57
# @Author  : young
# @File    : keras_base.py
# @Software: PyCharm
from typing import List, Any

import os
import numpy as np
import tensorflow as tf
from datetime import datetime

import keras.backend as K
from keras.callbacks import Callback
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adamax, Adam, Nadam
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class Metrics(Callback):
    """
    自定义评估函数
    """
    val_f1s: List[Any]
    val_recalls: List[Any]
    val_precisions: List[Any]

    def set_validation_data(self, validation_data):
        """
        赋予的评估函数自定义的评估用的数据
        :param validation_data: 验证集
                :example: X, y
        :return: 暂无
        """
        self.validation_data = validation_data

    def on_train_begin(self, logs=None):
        """
        训练开始前需要初始化的评估结果
        :param logs: 日志
                :type: dict
        :return: 暂无
        """
        if logs is None:
            logs = {}
        """
        True Positive(真正, TP)：将正类预测为正类数.
        True Negative(真负 , TN)：将负类预测为负类数.
        False Positive(假正, FP)：将负类预测为正类数 →→ 误报 (Type I error).
        False Negative(假负 , FN)：将正类预测为负类数 →→ 漏报 (Type II error)
        """
        # 验证集F1 score = 2TP/(2TP+FP+FN)
        self.val_f1s = []
        # 验证集召回率(recalls) = TP/(TP+FN)
        self.val_recalls = []
        # 验证集精准度(precisions) = TP/(TP+FP)
        self.val_precisions = []
        # 验证集准确率(accuracy) = (TP+TN)/(TP+FN+FP+TN)

    def on_batch_end(self, batch, logs=None):
        """
        每batch训练后可执行的评估
        :param batch: 批次的数据
        :param logs: 日志
        :return: 暂无
        """
        if logs is None:
            logs = {}
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' - val_f1: %.4f - val_precision: %.4f - val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
        return


class KerasBase:
    """
    模型基类
    """
    def __init__(self):
        """
        需要初始化的参数
        """
        # Keras模型Model()
        self.model = None
        # Keras模型Model().load()
        self.predict_model = None
        # 初始化Keras的数据生成器
        self.data_generator = None
        # 初始化评估用的验证集数据生成器
        self.evaluation_data_generator = None
        # 验证集分割比例
        self.validation_split = 0.15
        # 分词工具
        self.tokenizer = None
        # 文本的最大长度
        self.max_len = 180
        # 目标损失
        self.loss = None
        # Keras自带的评估函数
        self.acc = None
        # 训练轮数
        self.epochs = 5
        # 批次的数量
        self.batch_size = 64
        # 学习率（预训练模型可以调高学习率）
        self.lr_ph = 1e-5
        # 优化方法
        self.optimizer = 'Original_Adam'
        # 初始化自定义的评估函数
        self.metrics = Metrics()
        # 训练是否随机打乱数据
        self.isShuffle = False
        # 是否打印训练信息
        self.verbose = 1

    def set_metrics(self, metrics):
        """
        设置或者覆盖自定义的评估函数
        :param metrics: 评估函数
                :type: Metrics
        :return: 暂无
        """
        self.metrics = metrics

    def set_data_generator(self, data_generator):
        """
        设置或者覆盖自定义的训练数据生成器
        :param data_generator: 训练数据生成器
                :type: Generator
        :return: 暂无
        """
        self.data_generator = data_generator

    def set_evaluation_data_generator(self, evaluation_data_generator):
        """
        设置或者覆盖自定义的评估用验证集数据生成器
        :param evaluation_data_generator: 评估用验证集数据生成器
                :type: Method Object
        :return: 暂无
        """
        self.evaluation_data_generator = evaluation_data_generator

    def get_data(self, path):
        """
        读取训练数据
        :param path: 训练数据的路径
        :return: 暂无
        """
        pass

    def build_graph(self):
        """
        生成整个模型的图结构
        :return: 暂无
        """
        self._add_placeholders()
        print("successfully added placeholders")
        self._lookup_layer_op()
        print("successfully added an embedding layer")
        self._model_layer()
        print("successfully added an model")

    def _add_placeholders(self):
        """
        主要定义和初始化模型的输入Input()
        :return: 暂无
        """
        pass

    def _lookup_layer_op(self):
        """
        embedding层或者预训练模型层
        :return: 暂无
        """
        pass

    def _model_layer(self):
        """
        定义模型结构
        :return: 暂无
        """
        pass

    def train_by_fit_generator(self, train_data, dev_data, save_model_path):
        """
        执行训练, 所有参数可定制化
        :param train_data: 训练数据
        :param dev_data: 验证数据
        :param save_model_path: 模型保存的路径
        :return: 暂无
        """
        # 训练数据生成器
        train_d = self.data_generator(train_data, batch_size=self.batch_size, max_len=self.max_len)
        # 验证数据生成器
        valid_d = self.data_generator(dev_data, batch_size=self.batch_size, max_len=self.max_len)

        # 设置评估用的验证数据
        self.metrics.set_validation_data(self.evaluation_data_generator(self.tokenizer, dev_data))

        # 构建模型结构
        self.build_graph()

        # 设置模型损失，优化方法，自带的评估函数
        self.model.compile(
            loss=self.loss,
            optimizer=self.select_optimize(),
            metrics=[self.acc],
        )

        # 模型接入数据生成器，开始训练
        self.model.fit_generator(
            train_d.__iter__(self.tokenizer),
            steps_per_epoch=len(train_d),
            epochs=self.epochs,
            validation_data=valid_d.__iter__(self.tokenizer),
            validation_steps=len(valid_d),
            callbacks=[self.metrics]
        )

        # 保留模型
        self.model.save(save_model_path)

    def train_by_fit(self, train_data_path, dev_data_path, save_model_path):
        """
        执行训练, 所有参数可定制化
        :param train_data_path: 训练数据路径
        :param dev_data_path: 验证数据路径
        :param save_model_path: 模型保存的路径
        :return: 暂无
        """
        # 训练数据生成
        train_x, train_y = self.data_generator(train_data_path)
        # 验证数据生成
        valid_x, valid_y = self.data_generator(dev_data_path)

        # 设置评估用的验证数据
        # self.metrics.set_validation_data(self.evaluation_data_generator(self.tokenizer, dev_data))

        # 构建模型结构
        self.build_graph()

        # 设置模型损失，优化方法，自带的评估函数
        self.model.compile(
            loss=self.loss,
            optimizer=self.select_optimize(),
            metrics=[self.acc],
        )

        # 模型接入数据生成器，开始训练
        self.model.fit(
            x=train_x,
            y=train_y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=[valid_x, valid_y],
            verbose=self.verbose,
            shuffle=self.isShuffle
            # 待加入评估函数
        )

        # 保留模型
        self.model.save(save_model_path)

    def save(self, model_path):
        """
        保存模型（完整的结构与权重）
        :param model_path: 需要保留模型路径
        :return: 暂无
        """
        self.model.save(model_path)

    def save_weights_only(self, model_path):
        """
        保存模型（单权重）
        :param model_path:
        :return: 暂无
        """
        self.model.save_weights(model_path)

    def export_model(self,
                     export_model_dir,
                     model_version
                     ):
        """
        将模型保存为pb文件的格式
        :param export_model_dir: 需要储存pb模型文件目标文件目录
               :type: string
        :param model_version: 模型版本
               :type: int
        :return: 暂无
        """
        with tf.get_default_graph().as_default():
            # prediction_signature
            # input and output must be build_tensor_info
            tensor_info_input = tf.saved_model.utils.build_tensor_info(tf.convert_to_tensor(self.predict_model.input))
            tensor_info_output = tf.saved_model.utils.build_tensor_info(tf.convert_to_tensor(self.predict_model.output))
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': tensor_info_input},
                    outputs={'output': tensor_info_output},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
            print('step1 => prediction_signature created successfully')
            # set-up a builder
            export_path_base = export_model_dir
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(model_version)))
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            builder.add_meta_graph_and_variables(
                # tags:SERVING,TRAINING,EVAL,GPU,TPU
                sess=K.get_session(),
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={'prediction_signature': prediction_signature, },
            )
            print('step2 => Export path(%s) ready to export trained model' % export_path,
                  '\n starting to export model...')
            builder.save(as_text=True)
            print('Done exporting!')

    def load(self, model_path, custom_objects=None):
        """
        读取模型
        :param model_path: 模型的路径
        :param custom_objects: 自定的层对应的字典
                :type: dict
        :return:
        """
        start = datetime.now()
        if custom_objects:
            self.predict_model = load_model(model_path, custom_objects=custom_objects)
        else:
            self.predict_model = load_model(model_path)
        end = datetime.now()
        print("loading model costs: ", end - start)

    def predict(self, data):
        """
        inference的接口, 和下游任务强相关，所以在业务层实现
        :param data: 数据
        :return:
        """
        pass

    def print_model_summary(self):
        """
        打印模型的结构
        :return: 暂未与
        """
        print(self.model.summary())

    def select_optimize(self):
        """
        优化方法选择器
        :return: 暂无
        """
        if self.optimizer == 'Original_Adam':
            optimizer = Adam()
        elif self.optimizer == 'Adam':
            optimizer = Adam(lr=self.lr_ph, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(lr=self.lr_ph, rho=0.9, epsilon=None, decay=0.0)
        elif self.optimizer == 'Adadelta':
            optimizer = Adadelta(lr=self.lr_ph, rho=0.95, epsilon=None, decay=0.1)
        elif self.optimizer == 'Adagrad':
            optimizer = Adagrad(lr=self.lr_ph, epsilon=None, decay=0.0)
        elif self.optimizer == 'Adamax':
            optimizer = Adamax(lr=self.lr_ph, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00)
        elif self.optimizer == 'SGD':
            optimizer = SGD(lr=self.lr_ph, momentum=0.0, decay=0.0, nesterov=False)
        elif self.optimizer == 'Nadam':
            optimizer = Nadam(lr=self.lr_ph, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        else:
            optimizer = Adam(lr=self.lr_ph, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        return optimizer
