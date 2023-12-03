# YOLOv5 general utils

import glob
import logging
import math
import os
import random
import re
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from utils.metrics import fitness
from utils.torch_utils import init_torch_seeds

# Settings
# 控制print打印torch.tensor格式设置，tensor精度为5(小数点后5位)，每行字符数为320个，显示方法为long
torch.set_printoptions(linewidth=320, precision=5, profile='long')
# 控制print打印np.array格式设置，精度为5，每行字符数为320个
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
# pandas的最大显示行数为10
pd.options.display.max_columns = 10
# 阻止opencv参与多线程(与Pytorch的Dataloader不兼容)
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads  最大线程数


def set_logging(rank=-1):
    # 仅在sys.stderr打印日志消息，不写入文件
    # 打印logging.INFO及以上级别日志
    format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    logging.basicConfig(format=format, level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s     返回大于等于img_size且是s的最小倍数
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor     取大于等于x且是divisor的最小倍数
    return math.ceil(x / divisor) * divisor     # math.ceil 向上取整


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _     将特殊字符转为下划线
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """
    学习率衰减策略
    论文 https://arxiv.org/pdf/1803.09820.pdf:
    """
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = 'blue', 'bold', input[0]  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    """
    用在train.py中，得到每个类别的权重，频率高的权重低
    Get class weights (inverse frequency) from training labels
    :param labels: gt框数据
    :param nc: 类别数  2
    :return: torch.from_numpy(weights): 每个类别根据labels得到的占比(次数越多，权重越小) tensor
    """
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (94929, 5)     (labels_num, class index+xywh)
    # classes: 各个gt框的类别
    classes = labels[:, 0].astype(np.int)

    # np.bincount   https://blog.csdn.net/xlinsist/article/details/51346523
    # 统计classes中各个类别的个数     (1, nc)
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    weights[weights == 0] = 1  # 将出现次数为0的类别权重全部取1   replace empty bins with 1
    weights = 1 / weights  # 其他所有类别的权重，全部取次数的倒数     number of targets per class
    weights /= weights.sum()  # normalize   求出每一个类别的占比
    return torch.from_numpy(weights)    # numpy -> tensor


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """
    用在train.py中，利用上面得到的每个类别的权重，得到每一张图片的权重，再对图片进行按权重采样
    通过每张图片真实gt框的真实标签labels和上一步labels_to_class_weights得到的每个类别的权重进行采样     我们没用这里的处理逻辑......
    Produces image weights based on class_weights and image contents
    :param labels: 每张图片真实gt框的真实标签
    :param nc: 数据集的类别数 2
    :param class_weights: [2]   上一步labels_to_class_weights得到的每个类别的权重
    :return:
    """
    # class_counts: 每个类别出现的次数 [num_labels, nc] 每一行是当前这张图片每个类别出现的次数  num_labels=图片数量=label数量
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)    # 注释详见博客
    return image_weights


def xyxy2xywh(x):
    """
    将预测信息从xyxy格式转为xywh格式
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
    :param x: shape: (n, x1y1x2y2)   n: gt框的个数;  (x1, y1): gt框左上角;  (x2, y2): gt框右下角
    :return y: shape: (n, xywh)  n: gt框个数;  xy: gt框中心点;  wh: gt框宽高
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """
    用在test.py中，注意: x正方向为右面；y的正方向为下面
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    :param x: [n, xywh] (x, y)
    :return: y: [n, x1y1x2y2] (x1, y1): 左上角     (x2, y2): 右下角
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    用在datasets.py的 LoadImagesAndLabels类的__getitem__函数、load_mosaic、load_mosaic9等函数中
    将xywh(normalized) -> x1y1x2y2   (x, y): 中间点     wh: 宽高      (x1, y1): 左上点       (x2, y2): 右下点

    param:
        x: 某张图片的所有gt框的label信息的xywh部分(单张图片可能会存在多个gt框)。即xywh(xy为gt框中心点坐标的归一化值，wh为gt框宽高的归一化值)
        w: 该图片resize后的宽
        h: 该图片resize后的高
        padw、padw: (mosaic时: 子图边界与马赛克大图边界的距离) (letterbox时: 短边pad填充的尺寸大小)
    return:
        y: [top left x, top left y, bottom right x, bottom right y]，即该图片经resize后、pad裁剪后的，左上角点与右下角点的坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """
    用在datasets.py中的load_mosaic和load_mosaic9函数中
    xy(normalized) -> xy
    Convert normalized segments into pixel segments, shape (n,2)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    """
    用在datasets.py文件中的random_perspective函数中
    将一个多边形标签(不是矩形标签，到底是几边形未知)转化为一个矩形标签
    方法: 对多边形所有的点x1y1 x2y2 ... 获取其中的(x_min, y_min)和(x_max, y_max)作为矩形label的左上角和右下角
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    :param segment: 一个多边形标签 [n, 2] 传入这个多边形n个顶点的坐标
    :param width: 这个多边形所在图片的宽度
    :param height: 这个多边形所在图片的高度
    :return 矩形标签: [1, x_min + y_min + x_max + y_max]
    """
    # 分别获取当前多边形中所有多边形点的x和y坐标
    x, y = segment.T  # segment xy
    # inside: 筛选条件，xy坐标必须大于等于0，x坐标必须小于等于宽度，y坐标必须小于等于高度
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    # 获取筛选后的所有多边形点的x和y坐标
    x, y, = x[inside], y[inside]
    # 取当前多边形中xy坐标的最大最小值，得到边框的坐标xyxy
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    """
    用在datasets.py文件中的verify_image_label函数中
    将多个多边形标签(不是矩形标签，到底是几边形未知)转化为多个矩形标签
    Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    :param segments: [N, cls+x1y1+x2y2 ...]
    :return: [N, cls+xywh]
    """
    boxes = []
    for s in segments:
        # 分别获取当前多边形中所有多边形点的x和y坐标
        x, y = s.T  # segment xy
        # 取当前多边形中x和y坐标的最大最小值，得到边框的坐标xyxy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    # [N, cls+xywh]
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    用在datasets.py文件中的random_perspective函数中
    对segment重新采样，比如说segment坐标只有100个，通过interp函数将其采样为n个(默认1000)
    Up-sample an (n,2) segment
    :param segments: [N, x1x2...]
    :param n: 采样个数
    :return segments: [N, n/2, 2]
    """
    for i, s in enumerate(segments):
        # 0~len(s)-1 取n(1000)个点
        x = np.linspace(0, len(s) - 1, n)
        # 0,1,2,..., len(s)-1
        xp = np.arange(len(s))
        # 对所有的segments都进行重新采样，比如说segment坐标只有100个，通过interp函数将其采样为n个(默认1000)
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    # [N, n/2, 2]
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    用在detect.py和test.py中，将预测坐标从feature map映射回原图
    将坐标coords(x1y1x2y2)从img1_shape缩放到img0_shape尺寸
    Rescale coords (xyxy) from img1_shape to img0_shape
    :param img1_shape: coords相对于的shape大小
    :param coords: 要进行缩放的box坐标信息    x1y1x2y2    左上角 + 右下角
    :param img0_shape: 要将coords缩放到相对的目标shape大小
    :param ratio_pad: 缩放比例gain和pad值。None就先计算gain和pad值，然后再pad+scale，不为空就直接pad+scale
    """
    # calculate from img0_shape
    if ratio_pad is None:
        # gain  = old / new     取高宽缩放比例中较小的，之后还可以再pad。如果直接取大的，裁剪就可能减去目标
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        # wh padding    wh中有一个为0，主要是pad另一个
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain   # 缩放scale
    clip_coords(coords, img0_shape)     # 防止缩放后坐标过界，边界处直接剪掉
    return coords


def clip_coords(boxes, img_shape):
    """
    用在xyxy2xywhn、save_one_boxd等函数中
    将boxes坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(img_shape hw)内
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """
    # .clamp_(min, max) 将取值限定在(min, max)之间
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    在ComputeLoss的__call__函数中调用计算回归损失
    :param box1: 预测框
    :param box2: target人工标注框
    :return: box1和box2的Iou/GIou/DIou/CIou
    """
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes     获得边界框坐标
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area 交叉区域    tensor.clamp(0): 将矩阵中小于0的数变为0
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width 两个框的最小闭包区域的width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height 两个框的最小闭包区域的height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    """
    bbox_iou的简单版本，只计算iou
    https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4]) [N, x1y1x2y2]
        box2 (Tensor[M, 4]) [N, x1y1x2y2]
    Returns:
        box1和box2的面积
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn 求出box的面积
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)    # box1面积
    area2 = box_area(box2.T)    # box2面积

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # 等价于(torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)    # 各pred框与各gt框相交的面积
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


# def wh_iou(wh1, wh2):
#     # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
#     wh1 = wh1[:, None]  # [N,1,2]
#     wh2 = wh2[None]  # [1,M,2]
#     inter = torch.min(wh1, wh2).prod(2)  # [N,M]
#     return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """
    Runs Non-Maximum Suppression (NMS) on inference results
    Params:
        prediction: (bs, 3个Detect layer(feature map)的anchor_num * grid_w * grid_h堆叠在一起, xywh+conf+classes) = (64, 14553, 7)   所有feature map，3个anchor的预测结果总和
        conf_thres: 先进行一轮筛选，将分数过低的预测框(<conf_thres)删除(分数置0)
        iou_thres: iou阈值，如果“其余”预测框(iou第2/3/4/...大的预测框)，与target的iou > iou_thres，就将那个预测框置0。(非极大值抑制)
                        即，保留最优的预测框_1。对于次优的第2/3/4/...个预测框，若它们与最优预测框重叠过大，则它们就没有存在的意义了，就把他们消除掉。这就是NMS非极大值抑制的思想，仅保留最大值。
        classes: 是否nms后只保留特定的类别，默认None
        agnostic: 进行nms是否也去除不同类别之间的框，默认False
        multi_label: 是否是多标签 nc > 1    一般是True
        labels
    Returns:
        该函数处理一个batch的数据，返回一个list，每个元素是该batch中每张img的结果。一个batch有64张img，len(list) = batch_size = 64
            每张img的结果: (符合条件的预测框的个数, xyxy + 确定是物体的条件下，判定为某个class类的概率 + 该class类的index(0/1))，即(n, 6)
    """
    # Settings  设置一些变量
    nc = prediction.shape[2] - 5  # number of classes
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height   预测物体宽度和高度的大小范围 (min_wh, max_wh)
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()   每个图像最多检测物体的个数
    time_limit = 100.0  # seconds to quit after  nms执行时间阈值，超过这个时间就退出了    默认为10.0，这里我们自定义加大到100.0
    redundant = True  # require redundant detections    是否需要冗余的detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # batch_size个output，存放最终筛选后的预测框结果
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    # 定义第二层过滤条件
    xc = prediction[..., 4] > conf_thres  # candidates
    max_det = 300  # maximum number of detections per image     每张图片的最多目标个数
    merge = False  # use merge-NMS  多个bounding box给它们一个权重进行融合，默认False

    t = time.time()     # 当前时刻时间
    for xi, x in enumerate(prediction):
        # Apply constraints (源码中也注释了这层过滤)
        # 第一层过滤，滤除超小anchor和超大anchor
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height

        # 第二层过滤，根据conf_thres滤除背景目标(obj_conf<conf_thres的目标，置信度极低的目标)
        x = x[xc[xi]]  # confidence

        # {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        # Cat apriori labels if autolabelling 自动标注label时调用  一般不用
        # 自动标记在非常高的置信阈值（即 0.90 置信度）下效果最佳,而 mAP 计算依赖于非常低的置信阈值（即 0.001）来正确评估 PR 曲线下的区域。
        # 这个自动标注我觉得应该是一个类似RNN里面的Teacher Forcing的训练机制 就是在训练的时候跟着老师(ground truth)走
        # 但是这样又会造成一个问题: 一直靠老师带的孩子是走不远的 这样的模型因为依赖标签数据,在训练过程中,模型会有较好的效果
        # 但是在测试的时候因为不能得到ground truth的支持, 所以如果目前生成的序列在训练过程中有很大不同, 模型就会变得脆弱。
        # 所以个人认为(个人观点): 应该在下面使用的时候有选择的开启这个trick 比如设置一个概率p随机开启 或者在训练的前n个epoch使用 后面再关闭
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # 经过前两层过滤后如果该feature map没有目标框了，就结束这轮，直接进行下一张图片
        if not x.shape[0]:
            continue

        # Compute conf  计算conf_score
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2) 左上角 右下角
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            # 第三轮过滤: 针对每个类别score(obj_conf * cls_conf) > conf_thres
            # 一个框有多个类别的预测概率，需进行筛选
            # nonzero: 获得矩阵中的非0(True)数据的坐标  https://blog.csdn.net/qq_41780295/article/details/119725205
            # i: 符合条件的x的index       j: 类别index(0、1)
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T

            # box[i]: (2641, 4) xyxy    符合条件的x的左上角和右下角坐标
            # x: (2639, xyxy+conf+已知是物体的条件下，各class的概率)
            # x[i, j + 5, None]: (2641, 1)    对符合条件的x，取第(j+5)位置的值(已知是物体的条件下，第j个class的概率)
            # j[:, None]: (2641, 1)    class index
            # 最终的结果: (符合条件的x的个数, xyxy + 确定是物体的条件下，判定为某个class类的概率 + 该class类的index(0/1))
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)     # 一个类别直接取分数最大类的即可
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class   是否只保留特定的类别，默认None，不执行这里
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 检测数据是否为有限数。这轮可有可无，一般没什么用，所以这里他注释了
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes   如果经过第三轮过滤，该feature map没有目标框了，就结束这轮，直接进行下一张图片
            continue
        elif n > max_nms:  # excess boxes   如果经过第三轮过滤，该feature map还有很多框(>max_nms)，就需要排序
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS   第4轮过滤
        # 在这里，为了把各个class index区分开，我们对各个class index任意乘一个比较大的数(max_wh: 4096，就是顺便用了这个值而已，无实际意义)
        # 此时，class index = 0的类，值为0(0 * 4096 = 0)；class index = 1的类，值为4096(1 * 4096 = 4096)。因此，相较于0/1，这里变成了0/4096，区分度更大了
        c = x[:, 5:6] * (0 if agnostic else max_wh)

        # 做个切片，得到boxes和scores
        # 在class index为0的类上，x[:, :4]不变
        # 在class index为1的类上，x[:, :4]加max_wh(4096)
        # 在class index为2的类上，x[:, :4]加max_wh*2(4096*2)
        # ......
        # 在class index为n的类上，x[:, :4]加max_wh*n(4096*n)    我们这里是2分类，所以n只取0和1
        # 即在不同类别的box位置信息上，加一个很大但又不同的数(各类别的index*4096)，但各类别的scores保持不变
        # 这样作非极大抑制的时候，不同类别的框，大小会差的特别大，相当于类别隔离了，这样不同类别的框就不会掺和到一块了。
        #       通过这样的操作，本应该每个类别分别计算一次nms的问题，就可以直接把所有类别的nms一起给计算出来。这是一个做nms挺巧妙的技巧。
        #       https://blog.csdn.net/level_code/article/details/131245680
        boxes, scores = x[:, :4] + c, x[:, 4]
        # 返回经nms过滤后的bounding box(boxes)的索引(降序排列)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        if i.shape[0] > max_det:  # 限制每张图片的最大目标个数
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights 正比于 iou * scores
            # bounding box合并，其实就是把权重和框相乘再除以权重之和
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]   # 每个batch的最终输出 (n, 6)

        # 看下时间超没超时，超时没做完的就不做了
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    """
    用在train.py模型训练完后
    将optimizer、training_results、updates...从保存的模型文件f中删除
    Strip optimizer from 'f' to finalize training, optionally save as 's'
    :param f: 传入的原始保存的模型文件
    :param s: 删除optimizer等变量后的模型保存的地址 dir
    :return:
    """
    x = torch.load(f, map_location=torch.device('cpu'))     # 加载训练的模型
    # 如果模型是ema，replace model with ema
    if x.get('ema'):
        x['model'] = x['ema']
    # 以下模型训练涉及到的若干个指定变量置空
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1     # 模型epoch恢复初始值-1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)   # 保存模型 x -> s/f
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml'):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)


# 训练结果存储位置，自适应命名
def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def isdocker():
    # Is environment a Docker container
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False
