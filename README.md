# Aerial-Semantic-Segmentation-Drone
# dataset: https://www.kaggle.com/bulentsiyah/semantic-drone-dataset
# Multi-GPU:
           Example:
           config:{'strategy': 
           tf.distribute.MirroredStrategy(devices=[
                                                   "/gpu:0", 
                                                   "/gpu:1", 
                                                   "/gpu:2"])
           CUDA_VISIBLE_DEVICES="2,3,4" python train_v3.py

# Single-GPU:
           Example:
           CUDA_VISIBLE_DEVICES="2" python train.py 
           
# TODO：
           deeplabv3的tf.estimator下多GPU训练（改为MirroredStrategy）√
           复现的deeplabv3+在料口对齐上的效果 mIOU=0.9983 √
           MASK-RCNN源码（instance-segmentation）√
           增加图片强化：缩放、crop等
           预测、评估的代码
