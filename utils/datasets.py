# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str

# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

# Get orientation exif tag  相机设置
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


# 相机设置
def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      workers=2, prefix=''):
    """
    生成 dataloader, dataset
    自定义dataset类(LoadImagesAndLabels)，包括数据增强
    自定义dataloader：LoadImagesAndLabels(获取数据集) + DistributedSampler(分布式采集器) + InfiniteDataLoader(永久持续采样数据)
    :param path: 图片数据加载路径 train/test
    :param imgsz: train/test图片尺寸(数据增强后大小) 640
    :param batch_size: train: 32   val: 32*2=64
    :param stride: 模型最大stride=32    [32, 16, 8]
    :param hyp: 超参，这里主要用到里面一些关于数据增强(旋转、平移等)的系数
    :param augment: 是否要进行数据增强   train: True   val: False
    :param cache: 是否cache_images    False
    :param pad: 设置矩形训练的shape时进行的填充  默认0.0
    :param rect: 是否开启矩形train/test   train: False   val: True
    :param image_weights: 训练时是否根据图片样本真实框分布权重来选择图片   默认False
    :param prefix: 显示信息 一个标志，多为train/val，处理标签时保存cache文件会用到
    """
    dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                  augment=augment,
                                  hyp=hyp,
                                  rect=rect,
                                  cache_images=cache,
                                  single_cls=False,
                                  stride=int(stride),
                                  pad=pad,
                                  image_weights=False,
                                  prefix=prefix)

    batch_size = min(batch_size, len(dataset))

    # 使用InfiniteDataLoader和_RepeatSampler来对DataLoader进行封装, 代替原先的DataLoader, 能够永久持续的采样数据
    loader = InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=workers,
                        sampler=None,
                        pin_memory=True,    # 更快地将数据从CPU移动到GPU中
                        collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    当image_weights=False时(不根据图片样本真实框分布权重来选择图片)，就会调用InfiniteDataLoader和_RepeatSampler进行自定义DataLoader，
    进行持续性采样。
    Uses same syntax as vanilla DataLoader (使用与普通 DataLoader 相同的语法)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """
    Sampler that repeats forever
    这部分就是进行持续采样
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    """
    在detect.py中使用
    load文件夹中的图片/视频
    定义迭代器 用于detect.py
    """
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        # glob.glob: 返回所有匹配的文件路径列表      files: 提取图片所有路径
        if '*' in p:
            # 如果p是采样正则化表达式提取图片/视频，可使用glob获取文件路径
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            # 如果p是一个文件夹，使用glob获取全部文件路径
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            # 如果p是文件则直接获取
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]  # images: 目录下所有图片的图片名
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]  # videos: 目录下所有视频的视频名
        ni, nv = len(images), len(videos)   # 图片与视频的数量

        self.img_size = img_size
        self.stride = stride    # 最大下采样率
        self.files = images + videos    # 整合图片和视频路径到一个列表
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv    # 是不是video
        self.mode = 'image'     # 默认是读image模式
        if any(videos):
            # 判断有没有video文件，如果包含video文件，则初始化opencv中的视频模块，cap=cv2.VideoCapture等
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):     # 迭代器
        self.count = 0
        return self

    def __next__(self):     # 与iter一起用
        if self.count == self.nf:   # 数据读完了
            raise StopIteration
        path = self.files[self.count]   # 读取当前文件路径

        if self.video_flag[self.count]:     # 判断当前文件是否是视频
            # Read video
            self.mode = 'video'
            # 获取当前帧画面，ret_val为一个bool变量，直到视频读取完毕之前都是True
            ret_val, img0 = self.cap.read()
            # 如果当前视频读取结束，则读取下一个视频
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video     视频已经读取完了
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1     # 当前读取视频的帧数
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # 将图片缩放调整到指定大小，return img为缩放后的图片
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        # ::-1: BGR to RGB
        # transpose(2, 0, 1): HWC to CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)     # 将一个内存不连续存储的数组，转换为内存连续存储的数组，使得运行速度更快

        return path, img, img0, self.cap    # 返回文件路径，resize+pad后的图片，原始图片，视频对象

    def new_video(self, path):
        self.frame = 0  # 记录帧数
        self.cap = cv2.VideoCapture(path)   # 初始化视频对象
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 得到视频文件中的总帧数

    def __len__(self):
        return self.nf  # number of files


class LoadStreams:
    """
    load 文件夹中视频流
    multiple IP or RTSP camera
    定义迭代器 用于detect.py
    """
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride    # 最大下采样步长

        # 如果sources为一个保存了多个视频流的文件，获取每一个视频流，保存为一个列表
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            # 反之，只有一个视频流文件就直接保存
            sources = [sources]

        n = len(sources)    # 视频流个数
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later

        # 遍历每一个视频流
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            url = eval(s) if s.isnumeric() else s
            if 'youtube.com/' in str(url) or 'youtu.be/' in str(url):  # if source is YouTube video
                # check_requirements(('pafy', 'youtube_dl'))
                import pafy
                url = pafy.new(url).getbest(preftype="mp4").url
            # s = '0' Local webcam 本地摄像头
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # 视频宽
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 视频高
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100      # 获取视频帧率

            _, self.imgs[i] = cap.read()  # guarantee first frame   读取当前画面
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)   # 创建多线程读取视频流
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        # 获取进行resize+pad之后的shape，letterbox函数默认(参数auto=True)是按矩形推理进行填充的
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            # 警告：检测到不同的流尺寸。 为了获得最佳性能，请提供尺寸相同的流。
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()  # 用来指向下一帧
            if n == 4:  # 每4帧读取一次
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox: 将图片缩放调整到指定大小
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack     将读取的图片拼接在一起
        img = np.stack(img, 0)

        # Convert
        # ::-1: BGR to RGB
        # transpose(0, 3, 1, 2): BHWC to BCHW
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)     # 将一个内存不连续存储的数组，转换为内存连续存储的数组，使得运行速度更快

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    """
    根据imgs图片的路径找到对应labels的路径
    """
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=32, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size    # 经过数据增强后图片的大小。(数据集中图片的尺寸大小不一，我们要统一处理成self.img_size大小)
        self.augment = augment  # 是否启用数据增强，train: True  val:False
        self.hyp = hyp
        self.image_weights = image_weights  # 是否图片按权重采样，True就可以根据类别频率（频率越高权重越小，反之大）来进行采样。   默认False，不做类别区分
        self.rect = False if image_weights else rect    # 是否启动矩形训练，train: False  val: True，可以加速
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]   # 马赛克增强的边界值[-320, -320]
        self.stride = stride    # 最大下采样率 32
        self.path = path    # 图片路径

        # 得到path路径下的所有图片的路径self.img_files
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                # 获取数据集路径path，包含图片路径的txt文件或者包含图片的文件夹路径
                p = Path(p)
                # glob.glob 返回所有匹配的文件路径列表   递归获取p路径下所有文件
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            # 筛选f中所有的图片文件 (破折号替换为os.sep)
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee')

        # 根据imgs路径，找到labels的路径
        self.label_files = img2label_paths(self.img_files)  # 'VOCdevkit/labels/train/... .txt'

        # Check cache   下次运行这个脚本的时候，直接从cache中取label，而不是去文件中取label。速度更快
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            # 如果有cache文件，直接加载  exists=True: 从cache文件中读出了nf,nm,ne,nc,n等信息
            cache, exists = torch.load(cache_path), True
            # 如果图片版本信息或者文件列表的hash值对不上号，说明本地数据集图片和label可能发生了变化，重新cache label文件
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            # 否则，调用cache_labels缓存标签及标签相关信息
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:      # 如果已从cache文件读出标签信息，直接显示
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results

        # labels: 每张图片所对应的各行标注数据  (我们这里train和val都只有矩形gt，不存在多边形)
        # shapes: 每张图片的shape
        # self.segments: 如果数据集所有图片中没有一个多边形gt     self.segments为[]    (我们就是)
        #       否则存储数据集中所有存在多边形gt的图片的所有原始gt(肯定有多边形gt，也可能有矩形正常gt)
        # zip:  因为cache中所有labels，shapes，segments信息都是按每张图片img分开存储的，zip是将所有图片对应的信息叠在一起
        cache.pop('hash')  # remove hash    移除无用内容
        cache.pop('version')  # remove version  移除无用内容
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # 使用cache中的img_files信息，来赋值self.img_files
        self.label_files = img2label_paths(cache.keys())  # 用上面cache中的img_files信息，重新获取对应的label信息，然后赋值self.label_files

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # 每张图片分别属于哪一个batch_index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch_index of image
        self.n = n       # number of images
        self.indices = range(n)     # 每张图片分别对应的index

        # 为Rectangular Training做准备
        # 这里主要是shapes的生成，这一步很重要，因为如果采用矩形训练，那么每一个batch的内部，需采用统一的尺寸。因此需分别计算符合各个batch的各个shape
        # 对数据集按照高宽比进行排序，可保证同一个batch中图片形状差不多，这时再选一个该batch公用的shape，代价会比较小
        # Rectangular Training 矩形训练     train: False   val: True
        if self.rect:
            s = self.shapes     # 所有图片的shape  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio  高宽比
            irect = ar.argsort()    # 根据高宽比，对所有图片进行重排序
            self.img_files = [self.img_files[i] for i in irect]     # 获取排序后的img_files(排序后的图片路径)
            self.label_files = [self.label_files[i] for i in irect]  # 获取排序后的label_files(排序后的图片label路径)
            self.labels = [self.labels[i] for i in irect]   # 对labels进行排序 (labels中的各个元素，是每张图片所对应的各行标注数据(各个gt数据))
            self.shapes = s[irect]  # wh    获取排序后的wh
            ar = ar[irect]      # 获取排序后的aspect ratio  高宽比

            # Set training image shapes  每一个batch的内部，需采用统一的尺寸
            shapes = [[1, 1]] * nb      # nb: number of batches
            for i in range(nb):
                ari = ar[bi == i]   # bi: batch_index
                mini, maxi = ari.min(), ari.max()   # 获取第i个batch中，最小和最大高宽比
                # 第i个batch中，如果所有的高宽比都小于1(h < w)，则在后面将这个batch中所有图片的w设为img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]   # maxi: h相对指定尺度的比例      1: w相对指定尺度的比例
                # 第i个batch中，如果所有的高宽比都大于1(h > w)，则在后面将这个batch中所有图片的h设为img_size
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            # 计算矩形训练时，每个batch输入网络的shape值(向上取32的整数倍)
            # 要求每个batch的shape，高宽都是32的整数倍，所以要先除以32，取整再乘以32（不过img_size如果是32倍数，这里就没必要）
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride     # np.ceil 向上取整
        self.imgs = [None] * n      # 不对img图片进行缓存，太占空间了

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """
        加载label信息，生成cache文件
        :param path: cache文件保存地址
        :param prefix: 日志头部信息(彩打高亮部分)
        :return x: cache保存的字典，包括的信息有:
                        x[im_file]=[l, shape, segments]     # im_file为图片路径；l为各行的标注数据；shape为图片width、height；segments为多边形信息
                        hash: 利用所有img_files与所有label_files，生成的一个hash值
                        results: 统计所有img_files中，找到的label个数nf，丢失的label个数nm，空的label个数ne，破损的label个数nc，总img或label的个数len(self.img_files)
                                        注意，一张img图片可能有多个gt框，每个gt框占.txt中的一行。一张img图片中所有gt框数据属于一个label，而不是多个label。
                        version: 当前cache version (这里固定写死0.1)
        """
        # Cache dataset labels, check images and read shapes
        x = {}  # dict     初始化最终cache中保存的字典dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number of missing, number of found, number of empty, number of corrupted
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        # im_file: 当前这张图片的相对路径
        # lb_file: 当前这张图片的label的相对路径
        # l: [gt_num, cls+xywh(normalized)]
        #    如果这张图片没有一个segment多边形标签 l就存储原label(全部是正常矩形标签)
        #    如果这张图片有一个segment多边形标签  l就存储经过segments2boxes处理好的标签(正常矩形标签不处理 多边形标签转化为矩形标签)
        # shape: 当前这张图片的形状 shape
        # segments: 如果这张图片没有一个segment多边形标签 存储None
        #           如果这张图片有一个segment多边形标签 就把这张图片的所有gt存储到segments中(若干个正常gt 若干个多边形gt) [gt_num, xy1...]
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # 尝试读取整个图片文件，以检查其是否包含完整的、可以解码的图像数据
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]  # strip() 移除字符串首尾指定字符(默认移除空格或换行符)
                        if any([len(x) > 8 for x in l]):  # is segment 多边形  (我们这里，train训练集和val验证集，均未出现多边形数据)
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'  # 该图片存在两个或以上的gt框信息完全一致，属重复标签问题
                    else:
                        ne += 1  # label empty  该图片有对应的.txt标注文件，但该标注文件上没有任何一个gt框数据
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing    该图片没有对应的.txt标注文件
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]   # im_file为图片路径；l为各行的标注数据；shape为图片width、height；segments为多边形信息
            except Exception as e:
                nc += 1     # 数据异常，损坏的数据
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')     # corrupted 损坏的

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}.')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        里面包含了数据增强函数。
        训练 数据增强: moscic(random_perspective 随机视角) + hsv + 上下左右翻转
        测试 数据增强: letterbox
        :return torch.from_numpy(img): 这个index的图片，数据增强后的结果。  shape: (3,640,640)
        :return labels_out: 这个index的图片的gt框信息。 shape: (6, 6) = (gt_num, "0"+cls+xywh(normalized))    gt_num: gt框数量;  0: 就是数值0，一个初始化的值，后面再进行赋值;  cls: 类别信息，0:hat, 1:person;  xywh(normalized): 归一化后的坐标，xy为中心点，wh为宽高
        :return self.img_files[index]: 这个index的图片的路径地址
        :return shapes:  Train时为None    fro COCO mAP rescaling   暂不清楚用处
        """
        index = self.indices[index]     # self.indices 所有img图片的index
        hyp = self.hyp

        # self.mosaic，依照上面的结果，train: True; val: False
        # mosaic增强 对图像进行4张图拼接训练 一般训练时运行
        mosaic = self.mosaic and random.random() < hyp['mosaic']    # random.random() 随机生成[0, 1)的浮点数

        # mosaic + MixUp
        if mosaic:
            # img: 将4张图片拼接在一起(mosaic大图)，然后进行随机变换(数据增强)，最后rezise成一张标准大小的图片。shape: ndarray (640, 640, 3)
            # labels: img4对应的labels数据(同样经历了拼接、随机变换、resize的处理)。shape: ndarray (n, cls + x1y1x2y2)    n: 多个gt框; cls: 类别信息，0:hat, 1:person; x1y1x2y2: 分别为左上角、右下角的坐标
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            # 否则: 载入图片 + letterbox (val)
            # Load image
            # 载入图片，载入图片后还会进行一次resize，将当前图片的最长边缩放到指定的大小，较小边同比例缩放
            # img: resize后的图片    (h0, w0): 原始图片的hw      (h, w): resize后的图片hw
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            # letterbox之前确定这张当前图片letterbox之后的shape。如果不用self.rect矩形训练，shape就是self.img_size;
            # 如果使用self.rect矩形训练，shape就是对应batch的shape。因为矩形训练的话我们整个batch的shape必须统一。
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # val验证集，使用self.rect矩形训练，因此shape为对应batch的shape
            # letterbox: 将load_image缩放后的图片，再缩放到对应batch矩形训练时所约定的shape
            # (矩形推理需一个batch中所有图片的shape相同，各个batch的shape保存在self.batch_shapes中)
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # 图片letterbox之后label的坐标也要相应变化，根据pad调整label坐标，并将归一化的xywh -> 未归一化的xyxy
            labels = self.labels[index].copy()  # labels: ndarray(1, 5)  [cls(0:hat, 1:person) + xywh]
            # type(labels): numpy.ndarray，存在size属性
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            # 总结下，在val时这里主要做了三件事:
            #       1、load_image将图片从文件中加载出来，并resize到相应尺寸(最长边等于我们需要的尺寸，最短边等比例缩放)
            #       2、letterbox将1中resize后的图片再resize、pad到我们所需放到dataloader中的尺寸(collate_fn函数)。(矩形训练要求同一个batch中图片的尺寸必须保持一致)
            #       3、label的坐标同步处理。

        # self.augment，是否启用数据增强，train: True  val:False
        if self.augment:
            # mosaic，依照上面的结果，train: True; val: False
            if not mosaic:
                # 不做mosaic的话就要单独做random_perspective增强，因为mosaic函数内部执行了random_perspective增强
                # random_perspective增强: 随机对图片进行旋转，平移，缩放，截剪，透视变换
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # Augment colorspace  色域空间增强
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        nL = len(labels)  # gt框个数
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh  实际坐标
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1          坐标归一化
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1           坐标归一化

        # 平移增强，随机左右翻转 + 随机上下翻转
        # self.augment，是否启用数据增强，train: True  val:False
        if self.augment:
            # flip up-down  随机上下翻转
            if random.random() < hyp['flipud']:
                img = np.flipud(img)    # np.flipud 将数组在上下方向翻转
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]     # 1 - y_center  label也要映射

            # flip left-right   随机左右翻转
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)    # np.fliplr 将数组在左右方向翻转
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]     # 1 - x_center  label也要映射

        # 6个值的tensor，初始化gt框对应的图片序号，配合下面的collate_fn使用
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)    # numpy to tensor

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to shape: (3,640,640)
        img = np.ascontiguousarray(img)     # img变成内存连续的数据  加快运算

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        """
        这个函数在create_dataloader中生成dataloader时调用。将image和label整合到一起
        调用batch_size次getitem函数后，才会调用一次，对batch_size张图片和对应的label进行打包。
        :return torch.stack(img, dim=0): 堆叠整个batch的图片  shape: (32, 3, 640, 640)
        :return torch.cat(label, dim=0): 拼接整个batch的label  shape: (n, 6)，即 (num_target(整个batch中所有gt框的个数), img_index(该gt框归属于该batch中的那一张img) + class_index(0:hat, 1:person) + xywh(normalized))
        :return path: 整个batch所有图片的路径  len(path)=32
        :return shapes: len(shapes)=32，值全部为None
        pytorch的DataLoader打包一个batch的数据集时要经过此函数进行打包，通过重写此函数，实现标签与图片对应的划分，一个batch中哪些标签属于哪一张图片
        """
        # img: 一个tuple，由batch_size个tensor组成，整个batch中每个tensor表示一张图片
        # label: 一个tuple，由batch_size个tensor组成，每个tensor存放一张图片的所有gt框信息
        # path: 一个tuple，每个str对应一张图片的地址信息
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # l[:, 0]现为该batch中img的编号，batch size=32，所以取值0~31

        # torch.stack与torch.cat的区别      https://blog.csdn.net/weixin_37707670/article/details/119644333
        # torch.stack(img, 0): 将batch_size个(3, 640,640)拼成一个(batch_size, 3, 640, 640)
        # torch.cat(label, 0): 将(n0(img0的gt框个数), 6)、(n1(img1的gt框个数), 6)、(n2(img2的gt框个数), 6)...拼接成(n0+n1+n2+..., 6)
        # 之所以img与label的拼接方式不同，是因为img拼接时它的每个部分的形状是相同的，都是(3, 640, 640)
        # 而label的每个部分的形状是不同的，因为每张图片的gt框个数是不一样的。(label肯定也希望用stack，更方便，但是没办法这样拼)
        # 如果每张img的gt框个数都相同，那我们就可能不需要重写collate_fn函数了
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def load_image(self, index):
    """
    用在LoadImagesAndLabels模块的__getitem__函数和load_mosaic模块中
    从对应图片路径中载入对应index的图片，并将原图中hw中较大者扩展到self.img_size，较小者同比例扩展
    :param index: 当前图片的index
    :return:
        img: resize后的图片
        (h0, w0): 该图片原始的高宽
        img.shape[:2]: resize后，图片的高宽
    """
    # LoadImagesAndLabels中，self.imgs = [None] * n，因此img必为None。由于占用空间过大，我们不会通过self.imgs来缓存图片
    # img = self.imgs[index]

    # 从对应文件路径读出这张图片
    path = self.img_files[index]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # 该图片原始的高宽
    r = self.img_size / max(h0, w0)  # resize image to img_size     img_size是预处理后的输出图片尺寸，r为缩放比例
    if r != 1:
        # cv2.INTER_AREA: 基于区域像素关系的一种重采样或者插值方式，该方法是图像抽取的首选方法，它可以产生更少的波纹
        # cv2.INTER_LINEAR: 双线性插值，默认情况下使用该方式插值。     根据ratio选择不同的插值方式。
        interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    用在LoadImagesAndLabels模块的__getitem__函数
    hsv色域增强，处理图像hsv，不对label进行任何处理
    :param img: 待处理图片   BGR     (640, 640, 3)
    :param hgain: h通道色域参数，用于生成新的h通道
    :param sgain: s通道色域参数，用于生成新的s通道
    :param vgain: v通道色域参数，用于生成新的v通道
    :return:    返回hsv增强后的图片 img  (640, 640, 3)
    """
    # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数，用于生成新的hsv通道
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))     # 图像的通道拆分 h s v
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)          # 生成新的h通道
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)   # 生成新的s通道
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)   # 生成新的v通道

    # 图像的通道合并 img_hsv=h+s+v     随机调整hsv之后重新组合hsv通道
    # cv2.LUT(hue, lut_hue)     通道色域变换，输入变换前通道hue和变换后通道lut_hue
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def load_mosaic(self, index):
    """
    有名的mosaic增强模块，几乎训练的时候都会用到它，可以显著提高小样本的mAP
    用在LoadImagesAndLabels模块的__getitem__函数，进行mosaic数据增强
    将四张图片拼接在一张马赛克图像中    loads images in a 4-mosaic
    :param index: 需要获取的图像索引
    :return: img4: 将4张图片拼接在一起(mosaic大图)，然后进行随机变换(数据增强)，最后rezise成一张标准大小的图片。shape: ndarray (640, 640, 3)
             labels4: img4对应的labels数据(同样经历了拼接、随机变换、resize的处理)。shape: ndarray (n, cls + x1y1x2y2)    n: 多个gt框; cls: 类别信息，0:hat, 1:person; x1y1x2y2: 分别为左上角、右下角的坐标
    """
    # labels4: 用于存放拼接图像(4张图拼成一张)的label信息(不包含segments多边形)
    # segments: 用于存放拼接图像(4张图拼成一张)的label信息(包含segments多边形)
    labels4, segments4 = [], []
    s = self.img_size   # 处理后的图片大小
    # 随机初始化拼接图像的中心点坐标，[self.img_size*0.5 : self.img_size*1.5]之间，随机取2个数作为拼接图像的中心坐标
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center y, x
    # 从dataset中随机寻找额外的三张图像进行拼接
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices

    # 遍历四张图像进行拼接
    for i, index in enumerate(indices):

        # 每次取一张图片，并将这张图片的长边resize到self.size，短边同比例缩放
        img, _, (h, w) = load_image(self, index)    # img: resize后的结果图； _: 该图片原始高宽； (h, w): 该图片resize后的高宽

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 创建mosaic大图。当前mosaic大图shape: (1280, 1280, 3)
            # 下面以马赛克大图的左上子图(top_left)为例，进行解释
            # 先明确几个定义:
            #       h, w: 左上子图(top_left)的高宽尺寸
            #       xc, yc: 马赛克大图中心点的横纵坐标
            # 理清上述概念后，我们探究下面两个式子:

            #   x1a = max(xc - w, 0)
            #   y1a = max(yc - h, 0)
            #   x2a = xc
            #   y2a = yc
            #   上述内容为左上子图(top_left)位于马赛克大图上的坐标。左上角(x1a, y1a)， 右下角(x2a, y2a)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax

            # 第一种情况，若左上子图(top_left)较小，无需裁剪即可放进马赛克大图的左上方
            #   x1b = w - (x2a - x1a) = w - (xc - (xc - w)) = 0
            #   y1b = h - (y2a - y1a) = h - (yc - (yc - h)) = 0
            #   x2b = w
            #   y2b = h
            #   上述内容为左上子图(top_left)所使用的范围，坐标是相对于左上子图(top_left)本身的，左上角(x1b, y1b)，右下角(x2b, y2b)。这种情况左上子图(top_left)全用，未对其进行裁剪

            # 第二种情况，若左上子图(top_left)较大，放进马赛克大图时被裁剪了(这里假设左上子图高宽都较大，都超了，都被裁剪了)
            #   x1b = w - (x2a - x1a) = w - (xc - 0) = w - xc
            #   y1b = h - (y2a - y1a) = h - (yc - 0) = h - yc
            #   x2b = w
            #   y2b = h
            #   同为左上子图(top_left)所使用的范围，坐标是相对于左上子图(top_left)本身的，左上角(x1b, y1b)，右下角(x2b, y2b)。这种情况左上子图(top_left)未全用，对其进行了裁剪

            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将子图填充到mosaic大图的相对位置
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        # 计算当前子图边界与马赛克大图边界的距离。
        # pad值为正，则马赛克大图的当前位置可完全包含当前子图(当前子图较小，无需裁剪)
        # pad值为负，则马赛克大图的当前位置不能完全包含当前子图(当前子图较大，会被裁剪。pad值即为将要裁剪的大小)
        padw = x1a - x1b    # (自己画图感受)
        padh = y1a - y1b    # (自己画图感受)

        # Labels
        # labels: 获取对应拼接图像的所有正常label信息(如果有segments多边形会被转化为矩形label)      获取该拼接图像的所有gt框信息。shape: (gt_num, cls+xywh(normalized))  (一张图片可能有多个gt框，所以shape[0]=gt_num；cls为类别信息，0:hat, 1:person；xy为gt框中心点坐标的归一化值，wh为gt框宽高的归一化值)
        # segments: 获取对应拼接图像的所有不正常label信息(包含segments多边形也包含正常gt)     我们这里，train训练集和val验证集都没有多边形
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format，并且是该图片resize后、pad裁剪后的数据
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # 把4张图片的label信息，压缩在一起   labels4.shape: (n, 5)   # 4张图片共n个gt label，每个label维度是5
    labels4 = np.concatenate(labels4, 0)
    # 防止越界。label[:, 1:]中的所有元素的值(位置信息)必须在[0, 2*s]之间，小于0就令其等于0，大于2*s就令其等于2*s。     out: 返回
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

    # Augment 数据增强(图像随机变换)  shape: (1280, 1280, 3) => (640, 640, 3)
    # 对mosaic大图进行随机旋转，平移，缩放，裁剪，透视变换，并resize为单张标准图片的大小(img_size)
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    用在LoadImagesAndLabels模块的__getitem__函数
    将图片缩放调整到指定大小
    Resize and pad image while meeting stride-multiple constraints
    https://github.com/ultralytics/yolov3/issues/232
    :param img: 原图 hwc
    :param new_shape: 缩放后的大小
    :param color: pad的填充值
    :param auto: True 保证缩放后的图片保持原图的比例，即将原图较长边缩放到指定大小，再将原图较短边按原图比例缩放(不会失真)
                 False 将原图较长边缩放到指定大小，再将原图较短边按原图比例缩放，最后将较短边的两侧pad到较长边大小(不会失真)
    :param scaleFill: True 简单粗暴的将原图resize到指定的大小，相当于就是resize，没有pad操作(失真)
    :param scaleup: True 对于小于new_shape的原图进行缩放，大于的不变
                    False 对于大于new_shape的原图进行缩放，小于的不变    val验证集为False
    :return: img: letterbox后的图片 hwc
             ratio: wh ratios，缩放变化比率
            (dw, dh): w和h的pad，padding的尺寸大小
    """
    shape = img.shape[:2]  # letterbox操作前，img图片的shape  (height, width)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 只进行下采样(因上采样会让图片模糊)
    # only scale down, do not scale up (for better test mAP)    scaleup=False，对于大于new_shape (r<1)的原图进行缩放，小于new_shape (r>1) 的不变
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle   保证原图比例不变，将图像最大边缩放到指定大小
        # 取余操作可保证padding后的图片是stride(32)的整数倍
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch  简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 在较小边的两侧进行pad，而不是在一侧pad
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # shape:[h, w]   new_unpad:[w, h]
    if shape[::-1] != new_unpad:  # resize      将原图resize到new_unpad (长边相同，比例相同的新图)
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))    # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))    # 计算左右两侧的padding
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border，即add padding
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    """
    这个函数会用于load_mosaic中，用在mosaic操作之后。
    随机透视变换，对mosaic整合后的大图进行随机旋转、缩放、平移、裁剪、透视变换，最后resize到单张图片的标准尺寸(img_size=640)

    :param img: mosaic大图img4，shape: (2*img_size, 2*img_size, 3)
    如果mosaic大图没有一个多边形标签，则segments为空。(我们的train训练集和val验证集就是如此)
    :param targets: mosaic大图所有正常label标签，shanep: (n, cls+xyxy)   n个gt框; cls为类别信息，0:hat, 1:person; xyxy 经resize后、pad裁剪后，左上角点与右下角点的坐标
    :param segments: mosaic大图所有多边形label信息。(我们这里不涉及)
    :param degrees: 旋转参数
    :param translate: 平移参数
    :param scale: 缩放参数
    :param shear: 剪切参数
    :param perspective: 透视变换参数，透明度
    :param border: 用于确定最后输出图片的大小，一般为list: [-img_size, -img_size]。这样最后输出的图片大小为: (img_size, img_size)
    :return img: mosaic大图随机变换后的结果，shape: (img_size, img_size)
    :return targets: 经随机变换后的mosaic图的对应标签，shape: (n, cls+x1y1x2y2])，会消去边界外的gt框以及过小的gt框
    """

    height = img.shape[0] + border[0] * 2  # 640
    width = img.shape[1] + border[1] * 2    # 640

    # ==================================== 开始变换 ==========================================================
    # 注意，opencv是实现了仿射变换的，不过我们要先生成仿射变换矩阵M
    # Center 设置中心平移矩阵
    C = np.eye(3)   # 返回二维数组，对角线为1，其余为0
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective   设置透视变换矩阵
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale    设置旋转和缩放矩阵
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)   # 随机生成旋转角度
    s = random.uniform(1 - scale, 1 + scale)    # 随机生成旋转图像后的缩放比例
    # cv2.getRotationMatrix2D: 二维旋转缩放函数
    # 参数angle: 旋转角度     center: 旋转中心(默认就是图片中心)      scale: 旋转后图像的缩放比例
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear 设置裁剪矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation   设置平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined matrixs      @ 表示矩阵乘法，生成仿射变换矩阵M
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    # 将仿射变换矩阵M作用在图片上
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            # 透视变换函数，实现旋转平移缩放变换后的平行线不再平行
            # 参数和下面warpAffine类似
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            # 仿射变换函数，实现旋转平移缩放变换后的平行线依旧平行
            # image changed img [1280, 1280, 3] -> [640, 640, 3]
            # cv2.warpAffine: opencv实现的仿射变换函数
            # 参数: img: 需要变化的图像  M: 变换矩阵     dsize: 输出图像的大小      flags: 插值方法的组合(int 类型!)
            #       borderValue: (重点!)边界值填充，默认情况下为0
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates   对应调整标签信息
    n = len(targets)
    if n:
        # 判断是否使用segment标签: 只有segments不为空时即数据集中有多边形gt也有正常gt时才能使用segment标签 use_segments=True
        #       否则如果只有正常gt时segments为空 use_segments=False
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments   使用segments标签(标签中含有多边形gt)    我们这里train训练集和val验证集，都不存在多边形标签
            # 先对segment标签进行重采样
            # 比如说segment坐标只有100个，通过interp函数将其采样为1000个(默认1000)
            # [n, x1y2...x99y100] 扩增坐标 -> [n, 500, 2]
            # 由于有旋转，透视变换等操作，所以需要对多边形所有角点都进行变换
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):  # segment: [500, 2] 多边形的500个点坐标xy
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform  对该标签多边形的所有顶点坐标进行透视/仿射变换
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip  根据segment的坐标，取xy坐标的最大最小值，得到边框的坐标
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes     不使用segments标签，使用正常的矩形标签targets
            # 直接对box透视/仿射变换     由于有旋转，透视变换等操作，所以需对四个角点都进行变换
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform  每个角点的坐标
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip  去除太小的target(target大部分跑到图外去了)
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates     过滤target，筛选box
        # 长和宽必须大于wh_thr个像素，裁剪过小的框(面积小于裁剪前的area_thr)     长宽比范围在(1/ar_thr, ar_thr)之间的限制
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        # 得到所有满足条件的targets
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    """
    用在random_perspective中，对随机变换后的图片的label进行筛选。
    去除被裁剪过小的框(面积小于裁剪前的area_thr)，还有长和宽必须大于wh_thr个像素，且长宽比范围在(1/ar_thr, ar_thr)之间的限制.
    :param box1: [4, n]
    :param box2: [4, n]
    :param wh_thr: 筛选条件 宽高阈值
    :param ar_thr: 筛选条件 宽高比、高宽比最大值阈值
    :param area_thr:筛选条件 面积阈值
    :return i: 筛选结果 [n] 全是True或False    e.g: box1[i]即可得到i中所有等于True的矩形框，False的矩形框全部被删除
    """
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]   # 求出所有box1矩形框的宽和高 [n] [n]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]   # 求出所有box2矩形框的宽和高 [n] [n]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio   # 求出所有box2矩形框的宽高比和高宽比，取较大者 [n, 1]
    # 筛选条件: 增强后w、h要大于wh_thr；变换后的图像与变换前的图像面积比值大于area_thr；宽高比大于ar_thr
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)
