import logging
import numpy as np
import tensorflow as tf
import keras.backend as K

from random import random
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, UpSampling2D, Add
from keras.layers import Lambda, Concatenate
from keras.layers import Reshape, TimeDistributed, Dense, Conv2DTranspose
from keras.models import Model

from backbone.resnet import get_resnet
from models.mr_layers import ProposalLayer, PyramidROIAlign, DetectionLayer, DetectionTargetLayer
from utils import image_utils as utils
from utils.anchors import get_anchors


# ----------------------------------------------------------#
#   损失函数公式们
# ----------------------------------------------------------#
def batch_pack_graph(x, counts, num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def smooth_l1_loss(y_true, y_pred):
    """
    smmoth_l1 损失函数
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    建议框分类损失函数
    """
    # 在最后一维度添加一维度
    rpn_match = tf.squeeze(rpn_match, -1)
    # 获得正样本
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # 获得未被忽略的样本
    indices = tf.where(K.not_equal(rpn_match, 0))
    # 获得预测结果和实际结果
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # 计算二者之间的交叉熵
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    loss = K.switch(tf.math.is_nan(loss), tf.constant([0.0]), loss)
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """
    建议框回归损失
    """
    # 在最后一维度添加一维度
    rpn_match = K.squeeze(rpn_match, -1)

    # 获得正样本
    indices = tf.where(K.equal(rpn_match, 1))
    # 获得预测结果与实际结果
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)
    # 将目标边界框修剪为与rpn_bbox相同的长度。
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)
    # 计算smooth_l1损失函数
    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    loss = K.switch(tf.math.is_nan(loss), tf.constant([0.0]), loss)
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """
    classifier的分类损失函数
    """
    # 目标信息
    target_class_ids = tf.cast(target_class_ids, 'int64')
    # 预测信息
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    # 求二者交叉熵损失
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # 去除无用的损失
    loss = loss * pred_active

    # 求平均
    loss = tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(pred_active), 1)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """
    classifier的回归损失函数
    """
    # Reshape
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # 只有属于正样本的建议框用于训练
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # 获得对应预测结果与实际结果
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
    交叉熵损失
    """
    target_class_ids = K.reshape(target_class_ids, (-1,))
    # 实际结果
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))

    # 预测结果
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

    # 进行维度变换 [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # 只有正样本有效
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # 获得实际结果与预测结果
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


# ----------------------------------------------------------#
#   损失函数公式们
# ----------------------------------------------------------#
def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    # 载入图片和语义分割效果
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # print("\nbefore:",image_id,np.shape(mask),np.shape(class_ids))
    # 原始shape
    original_shape = image.shape
    # 获得新图片，原图片在新图片中的位置，变化的尺度，填充的情况等
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)
    # print("\nafter:",np.shape(mask),np.shape(class_ids))
    # print(np.shape(image),np.shape(mask))
    # 可以把图片进行翻转
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    if augmentation:
        import imgaug
        # 可用于图像增强
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        image_shape = image.shape
        mask_shape = mask.shape
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        mask = mask.astype(np.bool)
    # 检漏，防止某些层内部实际上不存在语义分割情况
    _idx = np.sum(mask, axis=(0, 1)) > 0

    # print("\nafterer:",np.shape(mask),np.shape(_idx))
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # 找到mask对应的box
    bbox = utils.extract_bboxes(mask)

    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # 生成Image_meta
    image_meta = utils.compose_image_meta(image_id, original_shape, image.shape,
                                          window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    # 1代表正样本
    # -1代表负样本
    # 0代表忽略
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # 创建该部分内容利用先验框和真实框进行编码
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    '''
    iscrowd=0的时候，表示这是一个单独的物体，轮廓用Polygon(多边形的点)表示，
    iscrowd=1的时候表示两个没有分开的物体，轮廓用RLE编码表示，比如说一张图片里面有三个人，
    一个人单独站一边，另外两个搂在一起（标注的时候距离太近分不开了），这个时候，
    单独的那个人的注释里面的iscrowing=0,segmentation用Polygon表示，
    而另外两个用放在同一个anatation的数组里面用一个segmention的RLE编码形式表示
    '''
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # 计算先验框和真实框的重合程度 [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # 1. 重合程度小于0.3则代表为负样本
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. 每个真实框重合度最大的先验框是正样本
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. 重合度大于0.7则代表为正样本
    rpn_match[anchor_iou_max >= 0.7] = 1

    # 正负样本平衡
    # 找到正样本的索引
    ids = np.where(rpn_match == 1)[0]
    # 如果大于(config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)则删掉一些
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # 找到负样本的索引
    ids = np.where(rpn_match == -1)[0]
    # 使得总数为config.RPN_TRAIN_ANCHORS_PER_IMAGE
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # 找到内部真实存在物体的先验框，进行编码
    ids = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]
        # 计算真实框的中心，高宽
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # 计算先验框中心，高宽
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w
        # 编码运算
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / np.maximum(a_h, 1),
            (gt_center_x - a_center_x) / np.maximum(a_w, 1),
            np.log(np.maximum(gt_h / np.maximum(a_h, 1), 1e-5)),
            np.log(np.maximum(gt_w / np.maximum(a_w, 1), 1e-5)),
        ]
        # 改变数量级
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bbox


# ------------------------------------#
#   五个不同大小的特征层会传入到
#   RPN当中，获得建议框
# ------------------------------------#
def rpn_graph(feature_map, anchors_per_location):
    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                    name='rpn_conv_shared')(feature_map)

    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
               activation='linear', name='rpn_class_raw')(shared)
    # batch_size,num_anchors,2
    # 代表这个先验框对应的类
    rpn_class_logits = Reshape([-1, 2])(x)

    rpn_probs = Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    x = Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
               activation='linear', name='rpn_bbox_pred')(shared)
    # batch_size,num_anchors,4
    # 这个先验框的调整参数
    rpn_bbox = Reshape([-1, 4])(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


# ------------------------------------#
#   建立建议框网络模型
#   RPN模型
# ------------------------------------#
def build_rpn_model(anchors_per_location, depth):
    input_feature_map = Input(shape=[None, None, depth],
                              name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location)
    return Model([input_feature_map], outputs, name="rpn_model")


# ------------------------------------#
#   建立classifier模型
#   这个模型的预测结果会调整建议框
#   获得最终的预测框
# ------------------------------------#
def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    # ROI Pooling，利用建议框在特征层上进行截取
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)

    # Shape: [batch, num_rois, 1, 1, fc_layers_size]，相当于两次全连接
    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                        name="mrcnn_class_conv1")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, 1, 1, fc_layers_size]
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)),
                        name="mrcnn_class_conv2")(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, fc_layers_size]
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                    name="pool_squeeze")(x)

    # Classifier head
    # 这个的预测结果代表这个先验框内部的物体的种类
    mrcnn_class_logits = TimeDistributed(Dense(num_classes),
                                         name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation("softmax"),
                                  name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # 这个的预测结果会对先验框进行调整
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
                        name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    mrcnn_bbox = Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    # ROI Pooling，利用建议框在特征层上进行截取
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                        name="mrcnn_mask_conv1")(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn1')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                        name="mrcnn_mask_conv2")(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                        name="mrcnn_mask_conv3")(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn3')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2D(256, (3, 3), padding="same"),
                        name="mrcnn_mask_conv4")(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn4')(x, training=train_bn)
    x = Activation('relu')(x)

    # Shape: [batch, num_rois, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE, channels]
    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                        name="mrcnn_mask_deconv")(x)
    # 反卷积后再次进行一个1x1卷积调整通道，使其最终数量为numclasses，代表分的类
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                        name="mrcnn_mask")(x)
    return x


def get_predict_model(config):
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

    # 输入进来的图片必须是2的6次方以上的倍数
    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    # meta包含了一些必要信息
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
    # 输入进来的先验框
    input_anchors = Input(shape=[None, 4], name="input_anchors")

    # 获得Resnet里的压缩程度不同的一些层
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)

    # 组合成特征金字塔的结构
    # P5长宽共压缩了5次
    # Height/32,Width/32,256
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    # P4长宽共压缩了4次
    # Height/16,Width/16,256
    P4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    # P4长宽共压缩了3次
    # Height/8,Width/8,256
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    # P4长宽共压缩了2次
    # Height/4,Width/4,256
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

    # 各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    # Height/4,Width/4,256
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    # Height/8,Width/8,256
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    # Height/16,Width/16,256
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    # Height/32,Width/32,256
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # 在建议框网络里面还有一个P6用于获取建议框
    # Height/64,Width/64,256
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # P2, P3, P4, P5, P6可以用于获取建议框
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # P2, P3, P4, P5用于获取mask信息
    mrcnn_feature_maps = [P2, P3, P4, P5]

    anchors = input_anchors
    # 建立RPN模型
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    rpn_class_logits, rpn_class, rpn_bbox = [], [], []

    # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    for p in rpn_feature_maps:
        logits, classes, bbox = rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)

    rpn_class_logits = Concatenate(axis=1, name="rpn_class_logits")(rpn_class_logits)
    rpn_class = Concatenate(axis=1, name="rpn_class")(rpn_class)
    rpn_bbox = Concatenate(axis=1, name="rpn_bbox")(rpn_bbox)

    # 此时获得的rpn_class_logits、rpn_class、rpn_bbox的维度是
    # rpn_class_logits : Batch_size, num_anchors, 2
    # rpn_class : Batch_size, num_anchors, 2
    # rpn_bbox : Batch_size, num_anchors, 4
    proposal_count = config.POST_NMS_ROIS_INFERENCE

    # Batch_size, proposal_count, 4
    # 对先验框进行解码
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])

    # 获得classifier的结果
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                             config.POOL_SIZE, config.NUM_CLASSES,
                             train_bn=config.TRAIN_BN,
                             fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

    detections = DetectionLayer(config, name="mrcnn_detection")(
        [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

    detection_boxes = Lambda(lambda x: x[..., :4])(detections)
    # 获得mask的结果
    mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES,
                                      train_bn=config.TRAIN_BN)

    # 作为输出
    model = Model([input_image, input_image_meta, input_anchors],
                  [detections, mrcnn_class, mrcnn_bbox,
                   mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                  name='mask_rcnn')
    return model


def get_train_model(config):
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

    # 输入进来的图片必须是2的6次方以上的倍数
    input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    # meta包含了一些必要信息
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

    # RPN建议框网络的真实框信息
    input_rpn_match = Input(
        shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
    input_rpn_bbox = Input(
        shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

    # 种类信息
    input_gt_class_ids = Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)

    # 框的位置信息
    input_gt_boxes = Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)

    # 标准化到0-1之间
    gt_boxes = Lambda(lambda x: utils.norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)

    # mask语义分析信息
    # [batch, height, width, MAX_GT_INSTANCES]
    if config.USE_MINI_MASK:
        input_gt_masks = Input(shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                               name="input_gt_masks", dtype=bool)
    else:
        input_gt_masks = Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], name="input_gt_masks",
                               dtype=bool)

    # 获得Resnet里的压缩程度不同的一些层
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)

    # 组合成特征金字塔的结构
    # P5长宽共压缩了5次
    # Height/32,Width/32,256
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    # P4长宽共压缩了4次
    # Height/16,Width/16,256
    P4 = Add(name="fpn_p4add")([
        UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    # P4长宽共压缩了3次
    # Height/8,Width/8,256
    P3 = Add(name="fpn_p3add")([
        UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    # P4长宽共压缩了2次
    # Height/4,Width/4,256
    P2 = Add(name="fpn_p2add")([
        UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])

    # 各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    # Height/4,Width/4,256
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    # Height/8,Width/8,256
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    # Height/16,Width/16,256
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    # Height/32,Width/32,256
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # 在建议框网络里面还有一个P6用于获取建议框
    # Height/64,Width/64,256
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # P2, P3, P4, P5, P6可以用于获取建议框
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # P2, P3, P4, P5用于获取mask信息
    mrcnn_feature_maps = [P2, P3, P4, P5]

    anchors = get_anchors(config, config.IMAGE_SHAPE)
    # 拓展anchors的shape，第一个维度拓展为batch_size
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
    # 将anchors转化成tensor的形式
    anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
    # 建立RPN模型
    rpn = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    rpn_class_logits, rpn_class, rpn_bbox = [], [], []

    # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
    for p in rpn_feature_maps:
        logits, classes, bbox = rpn([p])
        rpn_class_logits.append(logits)
        rpn_class.append(classes)
        rpn_bbox.append(bbox)

    rpn_class_logits = Concatenate(axis=1, name="rpn_class_logits")(rpn_class_logits)
    rpn_class = Concatenate(axis=1, name="rpn_class")(rpn_class)
    rpn_bbox = Concatenate(axis=1, name="rpn_bbox")(rpn_bbox)

    # 此时获得的rpn_class_logits、rpn_class、rpn_bbox的维度是
    # rpn_class_logits : Batch_size, num_anchors, 2
    # rpn_class : Batch_size, num_anchors, 2
    # rpn_bbox : Batch_size, num_anchors, 4
    proposal_count = config.POST_NMS_ROIS_TRAINING

    # Batch_size, proposal_count, 4
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])

    active_class_ids = Lambda(
        lambda x: utils.parse_image_meta_graph(x)["active_class_ids"]
    )(input_image_meta)

    if not config.USE_RPN_ROIS:
        # 使用外部输入的建议框
        input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                           name="input_roi", dtype=np.int32)
        # Normalize coordinates
        target_rois = Lambda(lambda x: utils.norm_boxes_graph(
            x, K.shape(input_image)[1:3]))(input_rois)
    else:
        # 利用预测到的建议框进行下一步的操作
        target_rois = rpn_rois

    """找到建议框的ground_truth
    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)]建议框
    gt_class_ids: [batch, MAX_GT_INSTANCES]每个真实框对应的类
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]真实框的位置
    gt_masks: [batch, height, width, MAX_GT_INSTANCES]真实框的语义分割情况
    Returns: 
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]内部真实存在目标的建议框
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]每个建议框对应的类
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]每个建议框应该有的调整参数
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]每个建议框语义分割情况
    """
    rois, target_class_ids, target_bbox, target_mask = \
        DetectionTargetLayer(config, name="proposal_targets")([
            target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

    # 找到合适的建议框的classifier预测结果
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                             config.POOL_SIZE, config.NUM_CLASSES,
                             train_bn=config.TRAIN_BN,
                             fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
    # 找到合适的建议框的mask预测结果
    mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES,
                                      train_bn=config.TRAIN_BN)

    output_rois = Lambda(lambda x: x * 1, name="output_rois")(rois)

    # Losses
    rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
        [input_rpn_match, rpn_class_logits])
    rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
        [input_rpn_bbox, input_rpn_match, rpn_bbox])
    class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
        [target_class_ids, mrcnn_class_logits, active_class_ids])
    bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
        [target_bbox, target_class_ids, mrcnn_bbox])
    mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
        [target_mask, target_class_ids, mrcnn_mask])

    # Model
    inputs = [input_image, input_image_meta,
              input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]

    if not config.USE_RPN_ROIS:
        inputs.append(input_rois)
    outputs = [rpn_class_logits, rpn_class, rpn_bbox,
               mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
               rpn_rois, output_rois,
               rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
    model = Model(inputs, outputs, name='mask_rcnn')
    return model
