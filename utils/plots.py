# Plotting utils

import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
matplotlib.rc('font', **{'size': 11})   # 自定义matplotlib图上字体font大小size=11
# 在PyCharm 页面中控制绘图显示与否
# 如果这句话放在import matplotlib.pyplot as plt之前就算加上plt.show()也不会再屏幕上绘图 放在之后其实没什么
matplotlib.use('Agg')  # for writing to files only


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    一般会用在detect.py中，在nms之后遍历每一个预测框，再将每个预测框画在原图
    使用opencv在原图img上画一个bounding box
    :param x: 预测得到的bounding box [x1, y1, x2, y2]
    :param img: 原图，要将bounding box画在这个图上  array
    :param color: bounding box线的颜色
    :param label: 标签上的框框信息，类别 + score
    :param line_thickness: bounding box的线宽
    """
    # tl = 框框的线宽，要么等于line_thickness，要么根据原图img长宽信息自适应生成一个
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # c1 = (x1, y1) = 矩形框的左上角   c2 = (x2, y2) = 矩形框的右下角
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
    # 注意: 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    # 如果label不为空还要在框框上面显示标签label + score
    if label:
        tf = max(tl - 1, 1)  # label字体的线宽 font thickness
        # cv2.getTextSize: 根据输入的label信息计算文本字符串的宽度和高度
        # 0: 文字字体类型  fontScale: 字体缩放系数  thickness: 字体笔画线宽
        # 返回retval 字体的宽高 (width, height), baseLine 相对于最底端文本的 y 坐标
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 同上面一样是个画框的步骤  但是线宽thickness=-1表示整个矩形都填充color颜色
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
        # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
        # [225, 255, 255]: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def output_to_target(output):
    """用在test.py中进行绘制前3个batch的预测框predictions 因为只有predictions需要修改格式 target是不需要修改格式的
    将经过nms后的output (num_obj，x1y1x2y2+conf+cls) -> (num_obj, batch_id+class+x+y+w+h+conf) 转变格式
    以便在plot_images中进行绘图 + 显示label
    Convert model output to target format (batch_id, class_id, x, y, w, h, conf)
    :params output: list{tensor(64)} 当前batch的64(batch_size)张img做完nms后的结果
                    list中每个tensor (n, 6)  n表示当前图片检测到的目标个数  6=x1y1x2y2+conf+cls
    :return np.array(targets): (num_targets, batch_id+class+xywh+conf)  其中num_targets为当前batch中所有检测到目标框的个数
    """
    targets = []
    for i, o in enumerate(output):  # 对每张图片分别处理
        for *box, conf, cls in o.cpu().numpy():     # 对每张图片的每个检测到的目标框进行convert格式
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    """
    将一个batch的图片都放在一个大图mosaic上面，放不下删除。
    用在test.py中进行绘制前3个batch的ground truth和预测框predictions(两个图) 一起保存 或者train.py中
    将整个batch的labels都画在这个batch的images上
    Plot image grid with labels
    :params images: 当前batch的所有图片  Tensor (batch_size, 3, h, w)  且图片都是归一化后的
    :params targets:  直接来自target: Tensor (target数量, img_index+class_index+xywh)
                      来自output_to_target: Tensor (num_pred, batch_id+class+xywh+conf) (num_pred, 7)
    :params paths: tuple  当前batch中所有图片的地址
                   如: 'VOCdevkit\\images\\train\\000000.jpg'
    :params fname: 最终保存的文件路径 + 名字  runs\train\exp\train_batch0.jpg
    :params names: 传入的类名，class index相应的key值  但是默认是None 只显示class index不显示类名
    :params max_size: 图片的最大尺寸640  如果images有图片的大小(w/h)大于640则需要resize 如果都是小于640则不需要resize
    :params max_subplots: 最大子图个数 16
    :return mosaic: 一张大图，最多可以显示max_subplots张图片，将图片(包括各自的label框)贴在一起显示。
                    mosaic每张图片的左上方还会显示当前图片的名字  最好以fname为名保存起来
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()   # tensor -> numpy array
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise  反归一化，将归一化后的图片还原
    if np.max(images[0]) <= 1:
        images *= 255

    # 设置一些基础变量
    tl = 3  # line thickness    设置线宽
    tf = max(tl - 1, 1)  # font thickness   设置字体笔画线宽
    bs, _, h, w = images.shape  # batch size(32), _ channel(3), height(640), width(640)
    bs = min(bs, max_subplots)  # limit plot images  子图总数，正方形
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)      ns = 每行/每列最大子图个数，子图总数=ns*ns ceil向上取整

    # Check if we should resize
    # 如果images有图片的大小(w/h)大于640则需要resize，如果都是小于640则不需要resize
    scale_factor = max_size / max(h, w)     # 1.0
    if scale_factor < 1:
        # 如果w/h有任何一条边超过640, 就要将较长边缩放到640, 另外一条边相应也缩放
        h = math.ceil(scale_factor * h)     # 512
        w = math.ceil(scale_factor * w)     # 512

    colors = color_list()  # list of colors

    # np.full 返回一个指定形状、类型和数值的数组
    # shape: (int(ns * h), int(ns * w), 3) (2560.0, 2560.0, 3)  填充的值: 255   dtype 填充类型: np.uint8
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    # 对batch内每张图片
    for i, img in enumerate(images):    # img (3, 640, 640)
        # 如果图片要超过max_subplots我们就不管了
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        # (block_x, block_y) 相当于是左上角的左边
        # 竖向理解，2 * 2
        block_x = int(w * (i // ns))    # 取整  0     0    640    640     ns=4.0
        block_y = int(h * (i % ns))     # 取余  0   640      0    640

        img = img.transpose(1, 2, 0)    # (640, 640, 3)  h w c
        if scale_factor < 1:    # 如果scale_factor < 1说明h/w超过max_size 需要resize回来
            img = cv2.resize(img, (w, h))

        # 将这个batch的图片一张张的贴到mosaic相应的位置上  hwc  这里最好自己画个图理解下
        # 第一张图mosaic[0:640, 0:640, :] 第二张图mosaic[640:1280, 0:640, :]
        # 第三张图mosaic[0:640, 640:1280, :] 第四张图mosaic[640:1280, 640:1280, :]
        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            # 求出属于这张img的target
            image_targets = targets[targets[:, 0] == i]
            # 将这张图片的所有target的xywh->xyxy
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            # 得到这张图片所有target的类别classes
            classes = image_targets[:, 1].astype('int')
            # 如果image_targets.shape[1] == 6，则说明没有置信度信息(此时target实际上是真实值)
            # 如果长度为7则第7个信息就是置信度信息(此时target为预测框信息)
            labels = image_targets.shape[1] == 6  # labels if no conf column
            # 得到当前这张图的所有target的置信度信息(pred) 如果没有就为空(真实label)
            # check for confidence presence (label vs pred)
            conf = None if labels else image_targets[:, 6]  # 注意，shape==7，即[0:6]

            if boxes.shape[1]:      # boxes.shape[1]不为空说明这张图有target目标
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    # 因为图片是反归一化的，所以这里boxes也反归一化
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:
                    # 如果scale_factor < 1，说明resize过，那么boxes也要相应变化
                    # absolute coords need scale if image scales
                    boxes *= scale_factor
            # 将img贴到mosaic上的位置计算
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y

            # 将当前img图片的boxes与标签，画到mosaic上
            for j, box in enumerate(boxes.T):
                # 遍历每个box
                cls = int(classes[j])   # 得到这个box的class index
                color = colors[cls % len(colors)]   # 得到这个box框线的颜色
                cls = names[cls] if names else cls  # 如果传入类名就显示类名 如果没传入类名就显示class index

                # 如果labels不为空说明是在显示真实target 不需要conf置信度 直接显示label即可
                # 如果conf[j] > 0.25 首先说明是在显示pred 且这个box的conf必须大于0.25 相当于又是一轮nms筛选 显示label + conf
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])    # 框框上面的显示信息
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)      # 一个个的画框

        # Draw image filename labels
        # 在mosaic每张图片相对位置的左上角写上每张图片的文件名，如 000000000315.jpg
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            # 返回文本 label 的宽高(width, height)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            # 要绘制的图像 +  要写上前的文本信息 + 文本左下角坐标 + 要使用的字体 + 字体缩放系数 + 字体的颜色 + 字体的线宽 + 矩形边框的线型
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        # mosaic内每张图片与图片之间弄一个边界框隔开 好看点 其实做法特简单 就是将每个img在mosaic中画个框即可
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    # 最后一步 check是否需要将mosaic图片保存起来。 fname: 保存路径
    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size   限制mosaic图片尺寸
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save  必须要numpy array -> tensor格式，才能保存
    return mosaic


def plot_study_txt(path='', x=None):  # from utils.plots import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    # ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [Path(path) / f'study_coco_{x}.txt' for x in ['yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        # for i in range(7):
        #     ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
        #     ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)


def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir=''):
    """
    用在训练结束，对训练结果进行可视化
    画出训练完的results.txt，Plot training 'results*.txt'，最终生成results.png
    :param start: 读取数据的开始epoch，因为result.txt的数据是一个epoch一行的
    :param stop: 读取数据的结束epoch。(如果为默认值0，函数中会再次处理)
    :param bucket: 是否需要从googleapis中下载results*.txt文件
    :param id: 需要从googleapis中下载的results + id.txt 默认为空
    """
    # 建造一个figure 分割成2行5列, 由10个小subplots组成
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()     # 将多维数组降为一维

    # s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
    #      'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']  # titles
    # 为了更好的进行说明，我们这里描述的更清晰一些。详细对应关系说明，见train.py开头，对results.txt表头的说明。
    # 我们这里只使用了results.txt中的如下10列数据。
    s = [
        'train_box_loss',
        'train_obj_loss',
        'train_cls_loss',
        'val_precision',
        'val_Recall',
        'val_box_loss',
        'val_obj_loss',
        'val_cls_loss',
        'val_mAP@.5',
        'val_mAP@.5:.95'
    ]
    if bucket:
        # files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
        files = ['results%g.txt' % x for x in id]
        c = ('gsutil cp ' + '%s ' * len(files) + '.') % tuple('gs://%s/results%g.txt' % (bucket, x) for x in id)    # cmd命令
        os.system(c)
    else:
        files = list(Path(save_dir).glob('results*.txt'))
    assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)

    # 读取files文件数据进行可视化
    for fi, f in enumerate(files):
        try:
            # skiprows=1，跳过第一行，即跳过表头
            # 如上所述，只使用results.txt其中的10列
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], skiprows=1, ndmin=2).T
            n = results.shape[1] - 1  # number of rows，except the title
            # 读取相应的轮次的数据
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):     # 分别可视化这10个指标
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values   loss值不能为0 要显示为np.nan
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else f.stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])   # 设置子图标题
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)


def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
    """
    通常用在train.py中，加载数据datasets和labels后，对labels进行可视化，分析labels信息
    plot dataset labels  生成labels_correlogram.jpg和labels.jpg   画出数据集的labels相关直方图信息
    :params labels: 数据集的全部真实框标签  (num_targets, class+xywh)  (94929, 5)
    :params names: 数据集的所有类别名
    :params save_dir: runs\train\exp
    :params loggers: 日志对象
    """
    print('Plotting labels... ')
    # c: classes(94929);    b: boxes xywh(4, 94929)
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes: 2
    colors = color_list()
    # pd.DataFrame: 创建DataFrame, 类似于一种excel, 表头是['x', 'y', 'width', 'height']  表格数据: b中数据按行依次存储
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    # 1、画出labels的xywh各自联合分布直方图  labels_correlogram.jpg
    # seaborn correlogram  seaborn.pairplot  多变量联合分布图: 查看两个或两个以上变量之间两两相互关系的可视化形式
    # data: 联合分布数据x   diag_kind:表示联合分布图中对角线图的类型   kind:表示联合分布图中非对角线图的类型
    # corner: True 表示只显示左下侧 因为左下和右上是重复的   plot_kws,diag_kws: 可以接受字典的参数，对图形进行微调
    sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    # 2、画出classes的各个类的分布直方图ax[0], 画出所有的真实框ax[1], 画出xy直方图ax[2], 画出wh直方图ax[3] labels.jpg
    matplotlib.use('svg')  # faster
    # 将整个figure分成2*2四个区域
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    # 第一个区域ax[1]画出classes的分布直方图
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')       # 设置y轴label
    if 0 < len(names) < 30:
        # 小于30个类别就把所有的类别名作为横坐标
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)      # 旋转90度 设置每个刻度标签
    else:
        # 如果类别数大于30个, 可能就放不下去了, 所以只显示x轴label
        ax[0].set_xlabel('classes')
    # 第三个区域ax[2]画出xy直方图     第四个区域ax[3]画出wh直方图
    sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    # 第二个区域ax[1]画出所有的真实框
    labels[:, 1:3] = 0.5  # center xy
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000     # xyxy
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)   # 初始化一个窗口
    for cls, *box in labels[:1000]:     # 把所有的框画在img窗口中
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors[int(cls) % 10])  # plot
    ax[1].imshow(img)
    ax[1].axis('off')   # 不要xy轴

    # 去掉上下左右坐标系(去掉上下左右边框)
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

    # loggers
    for k, v in loggers.items() or {}:
        if k == 'wandb' and v:
            # 这里我们commit设置为True，立即提交到wandb
            v.log({"Labels": [v.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=True)


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):  # from utils.plots import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')