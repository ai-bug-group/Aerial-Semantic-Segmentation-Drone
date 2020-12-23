# -*- coding: utf-8 -*-
# @Time    : 2020/12/23
# @Author  : young
# @File    : json_to_png.py
# @Software: PyCharm
import os
import cv2
import json
import os.path as osp
import numpy as np
from labelme.utils import image
from labelme.utils import labelme_shapes_to_label


def process(json_file, out_dir):
    """
    将label_me的数据转换为mask灰度图
    :param json_file: label_me生成的json文件目录
    :param out_dir:转换后png文件的目录
    :return:暂无
    """
    # 获取json文件列表(用glob直接会拿到绝对路径，不好取文件名字)
    json_file_list = os.listdir(json_file)
    for i in range(0, len(json_file_list)):
        # 获取每个json文件的绝对路径
        path = os.path.join(json_file, json_file_list[i])
        # 提取出.json前的字符作为文件名，以便后续保存Label图片的时候使用
        filename = json_file_list[i][:-5]
        # 拿到后缀
        extension = json_file_list[i][-4:]
        # 只要json文件
        if extension == 'json':
            if os.path.isfile(path):
                data = json.load(open(path))
                # 根据'imageData'字段的字符可以得到原图像
                img = image.img_b64_to_arr(data['imageData'])
                # data['shapes']是json文件中记录着标注的位置及label等信息的字段
                lbl, lbl_names = labelme_shapes_to_label(img.shape, data['shapes'])

                mask = []
                class_id = []
                # 跳过第一个class（因为0默认为背景,跳过不取！）
                for cls in range(1, len(lbl_names)):
                    # mask与class_id 对应记录保存
                    # 举例：当解析出像素值为1，此时对应第一个mask 为0、1组成的（0为背景，1为对象）
                    mask.append((lbl == cls).astype(np.uint8))
                    class_id.append(cls)

                mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])

                if not osp.exists(out_dir):
                    os.mkdir(out_dir)

                cv2.imwrite(osp.join(out_dir, '{}.png'.format(filename)), mask[:, :, 0])


if __name__ == '__main__':
    data_dir = "../data/zoomlion/train/json_files/"
    output_dir = "../data/zoomlion/train/masks/"
    process(data_dir, output_dir)
