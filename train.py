# -*- coding: utf-8 -*-
# @Time    : 2021/01/07
# @Author  : young
# @File    : train(v3).py
# @Software: PyCharm
import tensorflow as tf


from models.deeplab_v3_plus import DeeplabV3Plus
from tensorflow.keras.metrics import MeanIoU


class DistributedDataGenerator:
    """
    tf2.0+版本下多GPU模式已经不支持keras的fit_generator,
    改成与之前tf1.0+中estimator使用的tf.data.Dataset
    这种方式可以采用tf原生的图片读取，比使用cv2效率更高
    并且原生的支持tf_records
    """
    def __init__(self, configs):
        self.configs = configs
        self.assert_dataset()

    def assert_dataset(self):
        assert 'images' in self.configs and 'labels' in self.configs
        assert len(self.configs['images']) == len(self.configs['labels'])

    def __len__(self):
        return len(self.configs['images'])

    def read_img(self, image_path, is_mask=False):
        """
        通过tf.io和tf.image来读取原始图片和标注图片
        :param image_path:图片的地址(tensor)
                          -type:Tensor
        :param is_mask:是否是标注图
                       -type: boolean
        :return:
        """
        image = tf.io.read_file(image_path)
        if is_mask:
            image = tf.image.decode_png(image, channels=1)
            image.set_shape([None, None, 1])
            image = (tf.image.resize(
                images=image, size=[
                    self.configs['height'],
                    self.configs['width']
                ], method="nearest"
            ))
            # 将之前numpy的one_hot转成tf原生
            # (none, none, 1) -> (none, none) -> (none, none, 23)
            # 完美适配
            image = tf.squeeze(image, axis=-1)
            image = tf.one_hot(indices=image, depth=self.configs['num_classes'])
            image = tf.cast(image, tf.float32)
        else:
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = (tf.image.resize(images=image,
                                     size=[self.configs['height'],
                                           self.configs['width']
                                           ]
                                     ))
            image = tf.cast(image, tf.float32) / 127.5 - 1
        # todo 待更新数据增强
        return image

    def _map_function(self, image_list, mask_list):
        """
        读取原始图片和对应的标注图片路径列表
        :param image_list:[图片1，图片2，''']
                         -type: Tensor
        :param mask_list:[标注1，标注2，''']
                         -type: Tensor
        :return: image:[(none,none,3),(none,none,3),...]
                 mask:[(none,none,23),(none,none,23),...]
        """
        image = self.read_img(image_list)
        mask = self.read_img(mask_list, is_mask=True)
        return image, mask

    def get_dataset(self):
        """
        将数据转换为 tf.data.Dataset
        这里格式要说明一下:from_tensor_slices这个一般输入是（features，labels）
        features: [图片1，图片2，''']
        labels:   [标注1，标注2，''']
        :return: tf.data.Dataset
        """
        return tf.data.Dataset.from_tensor_slices((self.configs['images'],
                                                   self.configs['labels'])
                                                  )\
            .map(self._map_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .batch(self.configs['batch_size'], drop_remainder=True)\
            .repeat()\
            .prefetch(tf.data.experimental.AUTOTUNE)


class Trainer:
    """
    训练器，参数全部调整到config,
    Trainer实例化的时候将config(dict)传进去,例子如下
    config = {
        'name': 'aerial_semantic_deeplabv3+',
        'train_dataset_config': {
            'images': sorted(glob('data/train/images/*')),
            'labels': sorted(glob('data/train/masks/*')),
            'num_classes': 23, 'height': 300, 'width': 200, 'batch_size': 8
        },
        'val_dataset_config': {
            'images': sorted(glob('data/val/images/*')),
            'labels': sorted(glob('data/val/masks/*')),
            'num_classes': 23, 'height': 300, 'width': 200, 'batch_size': 8
        },
        'strategy': tf.distribute.MirroredStrategy(),
        'num_classes': 23, 'height': 300, 'width': 200,
        'backbone': 'mobilenetv2', 'learning_rate': 1e-4,
        'checkpoint_dir': 'checkpoints/deeplabv3-plus-aerial-semantic-mobilenetv2.h5',
        'epochs': 30
    }
    """
    def __init__(self, config):
        self.config = config
        self._assert_config()

        # load train and val data
        train_data_generator = DistributedDataGenerator(self.config['train_dataset_config'])
        self.train_data_length = len(train_data_generator)
        self.train_dataset = train_data_generator.get_dataset()
        print(str(self.train_data_length) + ' of train_data is loaded')

        val_data_generator = DistributedDataGenerator(self.config['val_dataset_config'])
        self.val_data_length = len(val_data_generator)
        self.val_dataset = val_data_generator.get_dataset()
        print(str(self.val_data_length) + ' of val_data is loaded')

        self._model = None

    @property
    def model(self):
        """
        build_model，与之前的区别就是需要加上tf.distribute.MirroredStrategy().scope()
        :return:
        """
        if self._model is not None:
            return self._model

        with self.config['strategy'].scope():
            self._model = DeeplabV3Plus(
                num_classes=self.config['num_classes'],
                backbone=self.config['backbone']
            )

            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config['learning_rate']
                ),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy',
                         MeanIoU(num_classes=self.config['num_classes'])]
            )
            return self._model

    @staticmethod
    def _assert_dataset_config(dataset_config):
        assert 'images' in dataset_config and \
            isinstance(dataset_config['images'], list)
        assert 'labels' in dataset_config and \
            isinstance(dataset_config['labels'], list)

        assert 'height' in dataset_config and \
            isinstance(dataset_config['height'], int)
        assert 'width' in dataset_config and \
            isinstance(dataset_config['width'], int)

        assert 'batch_size' in dataset_config and \
            isinstance(dataset_config['batch_size'], int)

    def _assert_config(self):
        assert 'train_dataset_config' in self.config
        Trainer._assert_dataset_config(self.config['train_dataset_config'])
        assert 'val_dataset_config' in self.config
        Trainer._assert_dataset_config(self.config['val_dataset_config'])

        assert 'strategy' in self.config and \
            isinstance(self.config['strategy'], tf.distribute.Strategy)

        assert 'num_classes' in self.config and \
            isinstance(self.config['num_classes'], int)
        assert 'backbone' in self.config and \
            isinstance(self.config['backbone'], str)

        assert 'learning_rate' in self.config and \
            isinstance(self.config['learning_rate'], float)

        assert 'checkpoint_dir' in self.config and \
            isinstance(self.config['checkpoint_dir'], str)

        assert 'epochs' in self.config and \
            isinstance(self.config['epochs'], int)

    def train(self):
        """
        模型训练，tf2.0+只保留fit模块了，同时可以传入之前格式data_generator,但不支持并行
        :return: history
        """
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config['checkpoint_dir'],
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                save_weights_only=True
            ),

        ]
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(str(self.train_data_length),
                                                                                   str(self.val_data_length),
                                                                                   self.config['train_dataset_config']
                                                                                   ['batch_size']))
        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,
            steps_per_epoch=self.train_data_length //
            self.config['train_dataset_config']['batch_size'],
            validation_steps=self.val_data_length //
            self.config['val_dataset_config']['batch_size'],
            epochs=self.config['epochs'], callbacks=callbacks
        )

        return history


if __name__ == "__main__":
    from glob import glob
    aerial_semantic_config = {
        'name': 'aerial_semantic_deeplabv3+',
        'train_dataset_config': {
            'images': sorted(glob('data/train/images/*')),
            'labels': sorted(glob('data/train/masks/*')),
            'num_classes': 23, 'height': 300, 'width': 200, 'batch_size': 8
        },
        'val_dataset_config': {
            'images': sorted(glob('data/val/images/*')),
            'labels': sorted(glob('data/val/masks/*')),
            'num_classes': 23, 'height': 300, 'width': 200, 'batch_size': 8
        },
        'strategy': tf.distribute.MirroredStrategy(),
        'num_classes': 23, 'height': 300, 'width': 200,
        'backbone': 'mobilenetv2', 'learning_rate': 1e-4,
        'checkpoint_dir': 'checkpoints/deeplabv3-plus-aerial-semantic-mobilenetv2.h5',
        'epochs': 30
    }

    trainer = Trainer(aerial_semantic_config)
    trainer.train()
