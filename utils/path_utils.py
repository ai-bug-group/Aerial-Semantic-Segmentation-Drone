# mobilenetv2 预训练模型
WEIGHTS_PATH_MOBILE_V2 = "https://github.com/bonlime/keras-deeplab-v3-plus/" \
                         "releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
# mask_rcnn coco预训练模型
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

# 类别数
num_classes = 2
# 数据目录
data_dir = '../data/zoomlion/train/'
# 生成tfrecords放置目录
tfrecord_path = '../data/zoomlion/train/records/'
# 图片目录
image_data_dir = 'images/'
# label目录，每一个像素点即为所分的类别
label_data_dir = 'masks/'

# 模型目录
model_dir = '../checkpoints'
# 是否清除模型目录
clean_model_dir = 'store_false'
# 训练epoch
train_epochs = 2
# 训练期间的验证次数
epochs_per_eval = 1

# tensorboard最大图片展示数
tensorboard_images_max_outputs = 6

# 批次设置
batch_size = 4
# 学习率衰减策略
learning_rate_policy = 'poly'
# 学习率衰减最大次数
max_iter = 30000

# 重载的结构
base_architecture = 'resnet_v2_101'
# 预训练模型位置
pre_trained_model = '../resnet_v2_101/resnet_v2_101.ckpt'
# 模型encoder输入与输出比例
output_stride = 16
# 是否更新BN参数
freeze_batch_norm = 'store_true'
# 起始学习率
initial_learning_rate = 7e-3
# 终止学习率
end_learning_rate = 1e-6
# global_step初始值
initial_global_step = 0
# 正则化权重
weight_decay = 2e-4

debug = None
