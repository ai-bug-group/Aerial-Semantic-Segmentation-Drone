import io
import os
from glob import glob
from PIL import Image
import tensorflow as tf
from utils import path_utils as config


# tfrecords转换的各种类型
def int_64_feature(value):
    return tf.train.Feature(int_64_feature=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_examples_list(path):
    """
    返回所有图片的index
    :param path:
    :return:
    """
    with tf.gfile.GFile(path) as f:
        lines = f.readlines()
    return [line.strip().split(' ')[0] for line in lines]


def main(split=0.1):
    """
    生成tfrecords
    :return:
    """
    if not os.path.exists(config.tfrecord_path):
        os.makedirs(config.tfrecord_path)
    # 相当于print
    tf.logging.info('读取数据')

    image_dir = os.path.join(config.data_dir, config.image_data_dir)
    label_dir = os.path.join(config.data_dir, config.label_data_dir)

    if not os.path.isdir(image_dir):
        raise ValueError('图片数据不存在，检查路径和数据')
    if not os.path.isdir(label_dir):
        raise ValueError('label数据不存在，检查路径和数据')
    # 获取训练和验证图片的名字
    train_examples = sorted(glob(image_dir+'*'))
    label_examples = sorted(glob(label_dir+'*'))
    num_val = int(len(train_examples) * split)
    num_train = len(train_examples) - num_val
    train_examples = list(zip(train_examples[:num_train], label_examples[:num_train]))
    val_examples = list(zip(train_examples[num_train:], label_examples[num_train:]))

    # 训练验证tfrecord存储地址
    train_output_path = os.path.join(config.tfrecord_path, 'train.record')
    val_output_path = os.path.join(config.tfrecord_path, 'val.record')

    # 生成tfrecord
    create_record(train_output_path, train_examples)
    create_record(val_output_path, val_examples)


def create_record(output_filename, examples):
    """
    将图片生成tfrecord
    :param output_filename: 输出地址
    :param examples: 图片和label
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 500 == 0:
            # 将生成第几张图片信息输出
            tf.logging.info('On image %d of %d', idx, len(examples))

        if not os.path.exists(example[0]):
            tf.logging.warning('没有该图片: ', example[0])
            continue
        elif not os.path.exists(example[1]):
            tf.logging.warning('没找着label文件： ', example[1])
            continue
        try:
            # 转换格式
            tf_example = dict_to_tf_example(example[0], example[1])

            writer.write(tf_example.SerializeToString())
        except ValueError:
            tf.logging.warning('无效的example： %s, 忽略', example)
    writer.close()


def dict_to_tf_example(image_path, label_path):
    """
    格式转换成tfrecord
    :param image_path: 输入图片地址
    :param label_path: 输出label地址
    :return:
    """
    print(image_path, label_path)
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoder_jpg = f.read()
    encoder_jpg_io = io.BytesIO(encoder_jpg)
    image = Image.open(encoder_jpg_io)

    if image.format != 'JPEG':
        tf.logging.info('输入图片格式错误')
        raise ValueError('输入图片格式错误')

    with tf.gfile.GFile(label_path, 'rb') as f:
        encoder_label = f.read()
    encoder_label_io = io.BytesIO(encoder_label)
    label = Image.open(encoder_label_io)

    if label.format != 'PNG':
        tf.logging.info('label图片格式错误')
        raise ValueError('label图片格式错误')

    if image.size != label.size:
        tf.logging.info('输入输出没对上')
        raise ValueError('输入输出没对上')

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(encoder_jpg),
        'label': bytes_feature(encoder_label)}))
    return example


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
