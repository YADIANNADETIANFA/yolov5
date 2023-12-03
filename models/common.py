# YOLOv5 common modules

import math
from copy import copy       # py也会区别浅拷贝和深拷贝
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp      # 混合精度训练模块

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized


def autopad(k, p=None):
    """
    用于Conv函数和Classify函数中
    根据卷积核大小k自动计算卷积核padding数(0填充)
    v5中只有两种卷积:
        1、下采样卷积: conv3*3 s=2 p=k//2=1
        2、feature size不变的卷积: conv1*1 s=1 p=k//2=1
    :param k: 卷积核的kernel_size
    :param p: 自动计算的需要pad的值(0填充)
    :return:
    """
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# def DWConv(c1, c2, k=1, s=1, act=True):
#     # Depthwise convolution
#     return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """
        在Focus、Bottleneck、BottleneckCSP、C3、SPP、DWConv、TransformerBloc等模块中调用
        Standard convolution    conv+BN+act
        :param c1: 输入channel值
        :param c2: 输出channel值
        :param k: 卷积核kernel_size
        :param s: 卷积stride
        :param p: 卷积padding     一般是None     可通过autopad自行计算需要pad的padding数
        :param g: 卷积的group数     =1就是普通卷积    >1就是深度可分离卷积
        :param act: 激活函数类型      True就是SiLU()/Swish      False就是不使用激活函数
                    类型是nn.Module就使用传进来的激活函数类型
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """
        用于Model类的fuse函数
        在前向传播的过程中，通过融合conv+bn，加速推理，一般用于测试/验证阶段
        """
        return self.act(self.conv(x))


# class TransformerLayer(nn.Module):
#     """
#     自注意力模块，详见博客  https://blog.csdn.net/qq_38253797/article/details/119684388
#     """
#     # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
#     def __init__(self, c, num_heads):
#         super().__init__()
#         self.q = nn.Linear(c, c, bias=False)    # 全连接层
#         self.k = nn.Linear(c, c, bias=False)
#         self.v = nn.Linear(c, c, bias=False)
#         self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
#         self.fc1 = nn.Linear(c, c, bias=False)
#         self.fc2 = nn.Linear(c, c, bias=False)
#
#     def forward(self, x):
#         x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
#         x = self.fc2(self.fc1(x)) + x
#         return x


# class TransformerBlock(nn.Module):
#     # Vision Transformer https://arxiv.org/abs/2010.11929
#     def __init__(self, c1, c2, num_heads, num_layers):
#         super().__init__()
#         self.conv = None
#         if c1 != c2:
#             self.conv = Conv(c1, c2)
#         self.linear = nn.Linear(c2, c2)  # learnable position embedding
#         self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
#         self.c2 = c2
#
#     def forward(self, x):
#         if self.conv is not None:
#             x = self.conv(x)
#         b, _, w, h = x.shape
#         p = x.flatten(2)
#         p = p.unsqueeze(0)
#         p = p.transpose(0, 3)
#         p = p.squeeze(3)
#         e = self.linear(p)
#         x = p + e
#
#         x = self.tr(x)
#         x = x.unsqueeze(3)
#         x = x.transpose(0, 3)
#         x = x.reshape(b, self.c2, w, h)
#         return x


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        """
        Standard bottleneck     Conv+Conv+shortcut
        :param c1: 第一个卷积的输入channel
        :param c2: 第二个卷积的输出channel
        :param shortcut: bool 是否shortcut连接，默认True
        :param g: 卷积分组的个数  =1就是普通卷积   >1就是深度可分离卷积
        :param e: expansion ratio  e*c2就是第一个卷积的输出channel，也就是第二个卷积的输入channel
        """
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class BottleneckCSP(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         """
#         在C3模块和yolo.py的parse_model模块调用
#         CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
#         :param c1: 整个BottleneckCSP的输入channel
#         :param c2: 整个BottleneckCSP的输出channel
#         :param n: 有n个Bottleneck
#         :param shortcut: bool Bottleneck中是否有shortcut，默认True
#         :param g: Bottleneck中的3*3卷积的g参数(Bottleneck内的第二个卷积的g参数)   =1普通卷积   >1深度可分离卷积
#         :param e: expansion ration c2*e=中间其他所有层的卷积核个数，即中间所有层的输入输出channel数
#         """
#         super(BottleneckCSP, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])     # 叠加n次Bottleneck
#
#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        """
        简化版的BottleneckCSP，因除了Bottleneck部分后只有三个卷积，故取名C3
        :param c1: 整个C3输入channel
        :param c2: 整个C3输出channel
        :param n:  有n个Bottleneck
        :param shortcut: bool Bottleneck中是否有shortcut，默认True
        :param g: Bottleneck中的3*3卷积的g参数(Bottleneck内的第二个卷积的g参数)   =1普通卷积   >1深度可分离卷积
        :param e: expansion ration，c2*e为中间其他所有层的卷积核个数，即中间所有层的输入输出channel数
        """
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# class C3TR(C3):
#     """
#     这部分是根据上面C3结构改编而来的，将原先的Bottleneck替换为调用TransformerBlock模块
#     """
#     # C3 module with TransformerBlock()
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)
#         self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        空间金字塔池化 Spatial pyramid pooling layer used in YOLOv3-SPP
        这个模块的主要目的是将更多不同分辨率的特征进行融合，得到更多的信息
        :param c1: SPP模块的输入channel
        :param c2: SPP模块的输出channel
        :param k: 保存着三个maxpool的窗口核大小，默认是(5, 9, 13)
        """
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)   # 第一层卷积
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)    # 最后一层卷积，+1是因为有len(k)+1个输入
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# class SPPF(nn.Module):
#     # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
#     def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#
#     def forward(self, x):
#         x = self.cv1(x)
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
#             y1 = self.m(x)
#             y2 = self.m(y1)
#             return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        """
        在yolo.py的parse_model函数中被调用
        理论: 从高分辨率图像中，周期性的抽出像素点重构到低分辨率图像中，即将图像相邻的四个位置进行堆叠，
            聚焦wh维度信息到c通道空间，提高每个点感受野，减少原始信息的丢失，该模块的设计主要是减少计算量加快速度，而不是增加网络的精度。
        Focus wh information into c-space
        先做4个slice，再concat，最后再Conv
        (b, c1, w, h)分成4个slice，每个slice(b, c1, w/2, h/2)
        concat(dim=1)4个slice，(b, c1, w/2, h/2) -> (b, 4c1, w/2, h/2)
        conv(b, 4c1, w/2, h/2) -> (b, c2, w/2, h/2)
        :param c1: slice前的channel
        :param c2: Focus最终输出的channel
        :param k: 卷积的kernel_size
        :param s: 卷积的stride
        :param p: 卷积的padding
        :param g: 最后卷积的分组情况  =1普通卷积   >1深度可分离卷积
        :param act: bool激活函数类型  默认True: SilU()/Swish  False: 不用激活函数
        """
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)   # 这里自定义的Conv类

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # ... 表示的含义     确定维度下的所有数据
        # https://zhuanlan.zhihu.com/p/264896206
        # https://blog.51cto.com/u_15088375/3248273
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


# class Contract(nn.Module):
#     """
#     用的不多
#     改变输入特征的shape，将w和h维度(缩小)的数据收缩到channel维度上(放大)
#     Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
#     """
#     def __init__(self, gain=2):
#         super().__init__()
#         self.gain = gain
#
#     def forward(self, x):
#         N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
#         s = self.gain
#         x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)     view: 改变tensor的维度
#         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)       permute: 改变tensor的维度顺序
#         return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


# class Expand(nn.Module):
#     """
#     用的不多
#     改变输入特征的shape，与Contract恰好相反，将channel维度(缩小)的数据扩展到w和h维度上(放大)
#     Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
#     """
#     def __init__(self, gain=2):
#         super().__init__()
#         self.gain = gain
#
#     def forward(self, x):
#         N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
#         s = self.gain
#         x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
#         x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
#         return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        """
        将 a list of tensors 按照某个维度进行concat，常用来合并前后两个feature map
        :param dimension: 沿着哪个维度进行concat
        """
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)     # x: a list of tensors


# class Detections:
#     """
#     用在AutoShape函数结尾     很少用
#     detections class for YOLOv5 inference results
#     """
#     def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
#         super(Detections, self).__init__()
#         d = pred[0].device  # device
#         gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
#         self.imgs = imgs  # list of images as numpy arrays
#         self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
#         self.names = names  # class names
#         self.files = files  # image filenames
#         self.xyxy = pred  # xyxy pixels
#         self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
#         self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
#         self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
#         self.n = len(self.pred)  # number of images (batch size)
#         self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
#         self.s = shape  # inference BCHW shape
#
#     def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
#         colors = color_list()
#         for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
#             str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
#             if pred is not None:
#                 for c in pred[:, -1].unique():
#                     n = (pred[:, -1] == c).sum()  # detections per class
#                     str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
#                 if show or save or render:
#                     for *box, conf, cls in pred:  # xyxy, confidence, class
#                         label = f'{self.names[int(cls)]} {conf:.2f}'
#                         plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
#             img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
#             if pprint:
#                 print(str.rstrip(', '))
#             if show:
#                 img.show(self.files[i])  # show
#             if save:
#                 f = self.files[i]
#                 img.save(Path(save_dir) / f)  # save
#                 print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
#             if render:
#                 self.imgs[i] = np.asarray(img)
#
#     def print(self):
#         self.display(pprint=True)  # print results
#         print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)
#
#     def show(self):
#         self.display(show=True)  # show results
#
#     def save(self, save_dir='runs/hub/exp'):
#         save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
#         Path(save_dir).mkdir(parents=True, exist_ok=True)
#         self.display(save=True, save_dir=save_dir)  # save results
#
#     def render(self):
#         self.display(render=True)  # render results
#         return self.imgs
#
#     def pandas(self):
#         # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
#         new = copy(self)  # return copy
#         ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
#         cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
#         for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
#             a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
#             setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
#         return new
#
#     def tolist(self):
#         # return a list of Detections objects, i.e. 'for result in results.tolist():'
#         x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
#         for d in x:
#             for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
#                 setattr(d, k, getattr(d, k)[0])  # pop out of list
#         return x
#
#     def __len__(self):
#         return self.n
