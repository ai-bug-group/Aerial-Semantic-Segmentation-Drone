# Aerial-Semantic-Segmentation-Drone
# dataset: https://www.kaggle.com/bulentsiyah/semantic-drone-dataset
# Multi-GPU:
           Example:
           config:{'strategy': 
           tf.distribute.MirroredStrategy(devices=[
                                                   "/gpu:2", 
                                                   "/gpu:3", 
                                                   "/gpu:4"])
           CUDA_VISIBLE_DEVICES="2,3,4" python train.py
           
# TODO：
           增加图片强化：缩放、crop等
           deeplabv3的tf.estimator下多GPU训练
           复现的deeplabv3+在料口对齐上的效果
           MASK-RCNN源码（instance-segmentation）
# MASK-RCNN的代码真的复杂啊=。=
# keras的多GPU在tf1.14.0和1.13.1上都有问题
