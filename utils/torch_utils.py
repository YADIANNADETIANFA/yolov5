# YOLOv5 PyTorch utils

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)


def init_torch_seeds(seed=0):
    """用在general.py的init_seeds函数
    用于初始化随机种子并确定训练模式
    Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    """
    # 为CPU设置随机种子，方便下次复现实验结果  to seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)
    # benchmark模式会自动寻找最优配置 但由于计算的随机性 每次网络进行前向传播时会有差异
    # 避免这种差异的方法就是将deterministic设置为True(表明每次卷积的高效算法相同)
    # 速度与可重复性之间的权衡  涉及底层卷积算法优化
    if seed == 0:
        # slower, more reproducible  慢 但是具有可重复性 适用于网络的输入数据在每次iteration都变化的话
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        # faster, less reproducible 快 但是不可重复  适用于网络的输入数据维度或类型上变化不大
        cudnn.benchmark, cudnn.deterministic = True, False


def time_synchronized():
    """这个函数被广泛的用于整个项目的各个文件中，只要涉及获取当前时间的操作，就需要调用这个函数
    精确计算当前时间  并返回当前时间
    https://blog.csdn.net/qq_23981335/article/details/105709273
    pytorch-accurate time
    先进行torch.cuda.synchronize()添加同步操作 再返回time.time()当前时间
    为什么不直接使用time.time()取时间，而要先执行同步操作，再取时间？说一下这样子做的原因:
       在pytorch里面，程序的执行都是异步的。
       如果time.time(), 测试的时间会很短，因为执行完end=time.time()程序就退出了
       而先加torch.cuda.synchronize()会先同步cuda的操作，等待gpu上的操作都完成了再继续运行end = time.time()
       这样子测试时间会准确一点
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    """
    用于train.py中载入预训练模型时，筛选预训练权重中的键值对
    用于筛选字典中的键值对，将da中的键值对复制给db，但是除了exclude中的键值对
    """
    # 返回字典da中的键值对  要求键k在字典db中且全部都不在exclude中 同时da中值的shape对应db中值的shape(相同)
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    """
    初始化模型权重
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:  # 如果是二维卷积就跳过，或者使用何凯明初始化
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:   # 如果是BN层，就设置相关参数: eps和momentum
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            # 如果是这几类激活函数 inplace插值就赋为True
            # inplace = True 指进行原地操作 对于上层网络传递下来的tensor直接进行修改 不需要另外赋值变量
            # 这样可以节省运算内存，不用多储存变量
            # inplace=True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            # https://www.cxyck.com/article/129380.html
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    """在yolo.py中Model类的fuse函数中调用
    融合卷积层和BN层(测试推理使用)   Fuse convolution and batchnorm layers
    方法: 卷积层还是正常定义, 但是卷积层的参数w,b要改变   通过只改变卷积参数, 达到CONV+BN的效果
          w = w_bn * w_conv   b = w_bn * b_conv + b_bn   (可以证明)
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    https://github.com/ultralytics/yolov3/issues/807
    https://zhuanlan.zhihu.com/p/94138640
    :params conv: torch支持的卷积层
    :params bn: torch支持的bn层
    """
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    # w_conv: 卷积层的w参数 直接clone conv的weight即可
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    # w_bn: bn层的w参数(可以自己推到公式)  torch.diag: 返回一个以input为对角线元素的2D/1D 方阵/张量?
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # w = w_bn * w_conv      torch.mm: 对两个矩阵相乘
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    # b_conv: 卷积层的b参数 如果不为None就直接读取conv.bias即可
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    # b_bn: bn层的b参数(可以自己推到公式)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # b = w_bn * b_conv + b_bn   (w_bn not forgot)
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    """
    用于yolo.py文件的Model类的info函数
    输出模型的所有信息 包括: 所有层数量, 模型总参数量, 需要求梯度的总参数量, img_size大小的model的浮点计算量GFLOPs
    :param model: 模型
    :param verbose: 是否输出每一层的参数parameters的相关信息
    :param img_size: int or list  i.e. img_size=640 or img_size=[640, 320]
    """
    # n_p: 模型model的总参数，number parameters
    n_p = sum(x.numel() for x in model.parameters())
    # n_g: 模型model的参数中需要求梯度(requires_grad=True)的参数量，number gradients
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)

    try:  # FLOPS
        from thop import profile    # 导入计算浮点计算量FLOPs的工具包
        # stride 模型的最大下采样率 有[8, 16, 32] 所以stride=32
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        # 模拟一样输入图片 shape=(1, 3, 32, 32)  全是0
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        # 调用profile计算输入图片img=(1, 3, 32, 32)时当前模型的浮点计算量GFLOPs   stride GFLOPs
        # profile求出来的浮点计算量是FLOPs  /1E9 => GFLOPs   *2是因为profile函数默认求的就是模型为float64时的浮点计算量
        # 而我们传入的模型一般都是float32 所以乘以2(可以点进profile看他定义的add_hooks函数)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        # expand  img_size -> [img_size, img_size]=[640, 640]
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        # 根据img=(1, 3, 32, 32)的浮点计算量flops推算出640x640的图片的浮点计算量GFLOPs
        # 不直接计算640x640的图片的浮点计算量GFLOPs可能是为了高效性吧, 这样算可能速度更快
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    # 添加日志信息
    # Model Summary: 所有层数量, 模型总参数量, 需要求梯度的总参数量, img_size大小的model的浮点计算量GFLOPs
    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def copy_attr(a, b, include=(), exclude=()):
    """在ModelEMA函数和yolo.py中Model类的autoshape函数中调用
    复制b的属性给a
    :params a: 对象a(待赋值)
    :params b: 对象b(赋值)
    :params include: 可以赋值的属性
    :params exclude: 不能赋值的属性
    """
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    # __dict__返回一个类的实例的属性和对应取值的字典
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            # 将对象b的属性k赋值给a
            setattr(a, k, v)


# 模型的指数加权平均(Model Exponential Moving Average)
# 利用滑动平均的参数来提高模型在测试数据上的健壮性/鲁棒性，用于测试阶段
# EMA模型也称为阴影模型(shadow)，其模型权重值更加平滑，泛化能力更好，避免了模型权重的大幅度震荡
class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, updates=0):
        """
        :param decay: 衰减函数参数，默认0.9999，考虑过去10000次的真实值
        :param updates: ema更新次数
        """
        # Create EMA 创建ema模型
        # eval():   https://zhuanlan.zhihu.com/p/357075502      不启用BN和Dropout
        self.ema = deepcopy(model).eval()
        self.updates = updates  # number of EMA updates，ema更新次数
        # self.decay: 衰减函数，输入变量为x
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        # 所有参数取消设置梯度(测试 model.val)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters     更新EMA模型的权重
        with torch.no_grad():
            self.updates += 1   # EMA模型更新次数 + 1
            d = self.decay(self.updates)    # 随着更新次数，更新参数贝塔(d)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    # EMA_t = (1- α) * EMA_t-1 + α * X_t
                    v *= d      # 对应 (1- α) * EMA_t-1 部分
                    v += (1. - d) * msd[k].detach()     # 对应 α * X_t 部分

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes     调用上面的copy_attr函数 从model中复制相关属性值到self.ema中
        copy_attr(self.ema, model, include, exclude)


"""
qry:
yolov5代码中，`ema.update(model)`与`ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])`有什么区别?

ans:
1. `ema.update(model)`: 这个调用主要是用于更新 EMA 模型的权重。通常，这个操作在每个训练批次（batch）后都会进行，以确保 EMA 模型随着原始模型的更新而更新。
    具体来说，这个函数会用原始模型的当前权重来更新 EMA 模型的权重。

2. `ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])`: 这个调用主要用于更新 EMA 模型的非权重属性。
    这包括模型的配置（如在 YAML 文件中定义的）、类别数（nc）、超参数（hyp）、类别名称（names）、卷积层的步长（stride）以及类别权重（class_weights）等。
    这个操作通常在每个 epoch 结束时进行一次，以确保 EMA 模型的这些属性与原始模型保持一致。
    
区别：
+ `ema.update(model)` 主要关注于权重的更新，而 ema.update_attr(...) 主要关注于非权重属性的更新。
+ `ema.update(model)` 通常更频繁地被调用（例如，在每个训练批次后），而 ema.update_attr(...) 通常只在每个训练周期（epoch）结束时被调用一次。
+ `ema.update(model)` 是用于保证 EMA 模型在预测性能上与原始模型相似，而 ema.update_attr(...) 是用于确保其他模型配置和属性与原始模型一致。

综上所述，这两个调用共同确保 EMA 模型在结构、配置和性能上都与原始模型保持高度一致。
"""