# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, initialize_weights, \
    copy_attr

try:
    import thop     # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    """
    Detect模块是用来构建Detect层的，将输入feature map通过一个卷积操作和公式，计算到我们想要的shape，为后面的计算损失或者NMS做准备
    """
    stride = None   # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=2, anchors=(), ch=()):   # detection layer
        """
        detection layer 相当于yolov3中的YOLOLayer层
        :param nc: number of classes  2
        :param anchors: 传入3个feature map上的所有anchor的大小    [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        :param ch: [128, 256, 512]  将要处理的3个feature map，分别的channel
        """
        super(Detect, self).__init__()
        self.nc = nc    # number of classes  2
        self.no = nc + 5    # number of outputs per anchor  classes+xywh+conf  7  实际使用顺序: xywh+conf+classes
        self.nl = len(anchors)      # number of detection layers    Detect的个数 3
        self.na = len(anchors[0]) // 2      # number of anchors     每个feature map的anchor个数 3
        self.grid = [torch.zeros(1)] * self.nl      # init grid  {list:3}   tensor()[0.] * 3
        # a shape:(3, 3, 2)   anchors以(w, h)对的形式存储  3个feature map，每个feature map上有3个anchor(w, h)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)

        # register_buffer
        # 模型中需要保存的参数一般有两种: 一种是反向传播需要被optimizer更新的，称为parameter；另一种不需要被更新，称为buffer
        # buffer的参数更新是在forward中，而optim.step只能更新nn.parameter类型的参数
        self.register_buffer('anchors', a)      # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))    # shape(nl, 1, na, 1, 1, 2)
        # output conv 对每个输出的feature map都要调用一次conv1*1
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv   input channel: 128/256/512  output channel: self.no * self.na=21

    def forward(self, x):
        """
        :return train: 一个tensor list，存放三个元素，(bs, anchor_num, grid_w, grid_h, xywh+conf+classes)
                        分别是(1, 3, 80, 80, 7), (1, 3, 40, 40, 7), (1, 3, 20, 20, 7)
                inference: 0: (64, 11088+2772+693, 7) = (bs, 3个Detect layer(feature map)的anchor_num * grid_w * grid_h堆叠在一起, xywh+conf+classes)
                           1: 一个tensor list，存放三个元素，(bs, anchor_num, grid_w, grid_h, xywh+conf+classes)
                                (64, 3, 80, 80, 7), (64, 3, 40, 40, 7), (64, 3, 20, 20, 7)
        """
        z = []      # inference output
        self.training |= self.export    # a |= b  《==》 a = a | b
        for i in range(self.nl):    # 对三个feature map分别进行处理
            x[i] = self.m[i](x[i])      # conv  xi(bs, 128/256/512, 80, 80) to (bs, 21, 80, 80)     21=3(anchors)*7(xywh+conf+classes)
            bs, _, ny, nx = x[i].shape
            # (bs, 21, 80, 80) to (bs, 3, 7, 80, 80) to (bs, 3, 80, 80, 7)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # contiguous()把tensor变成在内存中连续分布的形式

            # inference
            if not self.training:
                # 构造网格
                # 因为推理返回的不是归一化后的网格偏移量，需要再加上网格的位置，得到最终的推理坐标，再送入nms
                # 所以这里构造网格就是为了记录每个grid的网格坐标，方便后面使用
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()

                # 默认执行，不使用AWS Inferentia
                # 这里的公式和yolov3，v4中使用的不一样，是yolov5作者自己用的，效果更好
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]      # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]      # wh
                # z是一个tensor list，三个元素，分别是(64, 11088, 7) (64, 2772, 7) (64, 693, 7)    (bs, anchors * grid_w * grid_h, number of outputs per anchor)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """
        用于生成网格，主要用于生成相应的坐标
        输入的两个一维张量的元素个数分别是n1，n2
        输出的两个张量都是二维的，维度都是(n1, n2)
        输出的第一个张量，每行的元素值相同，不同行的值对应第一个输入张量
        输出的第二个张量，每列的元素值相同，不同列的值对应第二个输入张量
        https://blog.csdn.net/wsj_jerry521/article/details/126678226        有个图画错了，y坐标应该是[[0,0,0,0,0], [2,2,2,2,2], [3,3,3,3,3]]，它里面少了一维
        https://www.jianshu.com/p/50647b118db6
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s_hat.yaml', ch=3, nc=None, anchors=None):
        """
        :param cfg: 模型配置文件
        :param ch: input img channels 一般是3 RGB三通道
        :param nc: number of classes 数据集的类别个数   2
        :param anchors: 锚框信息

        因为我们这里仅对nc=2的本地数据集进行训练，并且也未对anchors框进行k-means重新计算，所以我们这里暂未使用`param nc`与`param anchors`这两个参数，
            如果有其他的训练任务，可关注下这两个参数，并对代码进行适当修改。
        """
        super(Model, self).__init__()
        import yaml  # for torch hub
        self.yaml_file = Path(cfg).name
        with open(cfg, encoding='UTF-8') as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)      # input channels    3

        # 创建网络模型
        # self.model: 初始化的整个网络模型(包括Detect层结构)
        # self.save: 所有层结构中from不等于-1的序号，并排好序 [4, 6, 10, 14, 17, 20, 23]
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])

        # default class names ['0', '1']
        self.names = [str(i) for i in range(self.yaml['nc'])]   # default names

        # Build strides, anchors
        # 获取Detect模块的stride(相对输入图像的下采样率)
        # 进而得出三个feature map下，分别anchor的信息。
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):   # True
            s = 256     # 2x min stride

            # 计算三个feature map的下采样率 [8, 16, 32]
            # torch.zeros(1, ch, s, s) 创建一张空白图片，空白图片输入通道数ch=3,图片尺寸s*s即256*256
            # 输入图片进行一次前馈传播，传播过程中，会在低层特征进行一次预测，在中层特征进行一次预测，在高层特征进行一次预测
            # 从网络结构图中我们可以分析出，低层预测时经历了3次卷积，图片大小缩了8倍，此时步长为8；中层预测又经历了一次卷积，图片大小又缩了一半，此时步长为16；高层预测又经历了一次卷积，图片大小又缩了一半，此时步长为32
            #   但是上述这些结论是我们从网络结构图中分析得出的，而代码是不知道的。所以这里用一张空白图片跑了一次网络，通过s / x.shape[-2]来计算出各层预测的步长
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # m.stride: tensor([8., 16., 32.])

            # view，维度重构     https://blog.csdn.net/york1996/article/details/81949843
            # 求出各个feature map下，分别anchor的大小
            m.anchors /= m.stride.view(-1, 1, 1)    # 原m.anchors，是各个anchor相对于原图的大小；m.anchors /= m.stride.view(-1, 1, 1)，是各个anchor相对各自feature map的大小

            check_anchor_order(m)   # 判断anchor顺序是否正确，即检查anchor顺序和stride顺序是否一致。无问题

            self.stride = m.stride
            self._initialize_biases()   # only run once     初始化偏置

        # Init weights, biases
        initialize_weights(self)    # 初始化模型权重

        # 输出模型的所有信息 包括: 所有层数量, 模型总参数量, 需要求梯度的总参数量, img_size大小的model的浮点计算量GFLOPs
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        return self.forward_once(x, profile)  # single-scale inference, train   正常向前推理

    # 前向传播，x为输入图片数据
    def forward_once(self, x, profile=False):
        """
        :param x: 输入图像
        :param profile: True 可以做一些性能评估
        :return train情况下: 一个tensor list，存放三个元素，[bs, anchor_num, grid_w, grid_h, xywh+conf+2classes]
                       分别是 (32, 3, 80, 80, 7) (32, 3, 40, 40, 7) (32, 3, 20, 20, 7)

                inference情况下: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+2classes]
                           1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+2classes]
                             [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]   ？没看懂，具体debug到这里再看
        """
        y = []  # outputs       y临时list，临时存储需保存的网络层输出。yolov5s_hat.yaml中Concat层的输入，就需要前面其他层的输出结果，这些层的结果就是临时存储在y中的
        for m in self.model:    # 各层网络
            # 前向推理每一层结构  m.i=index, m.f=from, m.type=类名, m.np=number of params
            if m.f != -1:   # if not from previous layer    该层的输入不是上一层的输出

                # 这里需要做4个concat操作和1个Detect操作
                # concat操作如m.f=[-1, 6]，x就有两个元素，一个是上一层的输出，另一个是index=6的层的输出，再送到x=m(x)做concat操作
                # Detect操作m.f=[17, 20, 23]，x有三个元素，分别存放第17层第20层第23层的输出，再送到x=m(x)做Detect的forward
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]   # from earlier layers

            x = m(x)    # run 正向推理   本层输入经过本层网络结构，获取本层输出

            # 存放着self.save的每一层的输出，因为后面需要用来做concat等操作要用到，不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)   # save output   只存需保存的网络层输出，其他用None占位即可

        # x[0].shape:(32, 3, 80, 80, 7)      batch size: 32, 3通道, feature map尺寸80*80/40*40/20*20
        # x[1].shape:(32, 3, 40, 40, 7)
        # x[2].shape:(32, 3, 20, 20, 7)
        # 7 = 5 + 2       5 = 4(anchor的坐标) + 1(置信度)   2为分类数
        return x    # 前向传播最终结果

    def _initialize_biases(self):  # initialize biases into Detect()
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):
            # b: Tensor(3, 7)
            # na: number of anchors 3;
            # 7 = (5 + 2);    5 = 4 + 1    4为检测框的坐标信息，1为检测框置信度信息，检测框存在目标的可能性大小
            # 2为类别数 'hat' 'person'
            b = mi.bias.view(m.na, -1)

            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99))
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):     # fuse model Conv2d() + BatchNorm2d() layers
        """
        用在detect.py、val.py
        fuse model Conv2d() + BatchNorm2d() layers
        调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
        """
        print('Fusing layers... ')
        # 遍历每一层结构
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构，那么就调用fuse_conv_and_bn函数将conv和bn进行融合，加速推理
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)     # update conv 融合
                delattr(m, 'bn')    # remove batchnorm  移除bn
                m.forward = m.fuseforward   # update forward    更新前向传播 (反向传播不用管，因为只用在推理阶段)
        self.info()     # 打印conv+bn融合后的模型信息
        return self

    def info(self, verbose=False, img_size=640):    # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):     # model_dict, input_channels(3)
    """
    用在上面Model模块中
    解析模型文件(字典形式)，并搭建网络结构。
    网络结构详见"yolov5_3.png"，无论是网络结构，还是输入输出数值，都对应得上。(除了最后Detect那里的75，因我们class为2，我们的话应该是21)
    各网络模块的详细细节，非常推荐参考 https://blog.csdn.net/qq_38253797/article/details/119684388 这里的图
    这个函数其实主要做的就是: 更新各层args(参数)，计算c2(各层输出channel) => 使用各层参数搭建网络 => 生成 layers + save
    :param d: model_dict 模型文件 字典形式 {dict: 7} yolov5s.yaml中的6个元素 + ch
    :param ch: 记录模型各层的输出channel，初始ch=[3] 后面会删除
    :return nn.Sequential(*layers): 网络每一层的层结构
    :return sorted(save): 把所有层结构中from不是-1的值记下，并排序[4, 6, 10, 14, 17, 20, 23]
    """
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']

    # na: 每一个predict head上的anchor数      number of anchors   =3
    na = len(anchors[0]) // 2

    # 每个feature map，每个grid的最终输出channel。(即predict head的最终输出)
    # 5 = 4 + 1     4为检测框的坐标信息；1为检测框的置信度信息(检测框存在目标的可能性大小)
    # no: number of outputs = anchors * (classes + 5)   3 * (2 + 5) = 21
    no = na * (nc + 5)

    # 开始搭建网络
    # layers: 存储网络搭建时的每一层
    # save: 记录下所有层结构中，from不是-1的层结构序号
    # c2: 输出通道数(保存当前层的输出channel)      c1表输入通道数，c2表输出通道数
    layers, save, c2 = [], [], ch[-1]
    # from(当前层输入来自哪些层), number(当前层次数，初定), module(当前层类别), args(当前层参数，初定)
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m)    # eval strings，解析字符串为对应class类，举例，eval('C3')，解析为models/common.py中的class C3类；eval('Detect')，解析为models/yolo.py中的class Detect类
        for j, a in enumerate(args):
            try:
                # False属bool，"isinstance(False, str)"结果为False
                # 'None'属str，"isinstance('None', str)"结果为True，eval(None)结果仍为None，不抛异常
                # eval('nearest')抛异常NameError: name 'nearest' is not defined
                # eval('nc')为2
                # eval('anchors')为[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:   # NameError: name 'nearest' is not defined
                pass

        # 更新当前层的args(参数)，调整模型深度(层数)、模型输出通道数
        # gd(depth_multiple)：调整模型深度(层数)，0.33     举例: yolov5s: n*0.33     n: 当层初定有n个模块
        # gw(width_multiple): 调整模型输出通道数，0.50
        # (仔细观察yolov5s.hat.yaml发现，仅C3模块的number(n)不是1；而且所谓的"该层模块的个数"，实际为"C3模块中bottleneck的个数")
        n = max(round(n * gd), 1) if n > 1 else n   # 获取该层模块个数，round四舍五入
        if m in [Conv, Bottleneck, SPP, Focus, C3]:     # DWConv、BottleneckCSP、C3TR、GhostConv、GhostBottleneck、MixConv2d、CrossConv
            # c1: 当前层的输入channel数；   c2: 当前层的输出channel数(初定)      ch: 记录着所有层的输出channel
            c1, c2 = ch[f], args[0]
            if c2 != no:    # if not output 非最终输出
                # 计算当前层的最终输出channel数（可以被8整除的值），8的倍数对GPU更加友好
                c2 = make_divisible(c2 * gw, 8)

            # 在初定arg的基础上更新，加入当前层的输入channel并更新当前层
            # [in_channel, out_channel, *args[1:]]
            args = [c1, c2, *args[1:]]
            # 如果当前层是C3，则需要在args中加入bottleneck的个数     # BottleneckCSP、C3TR
            # [in_channel, out_channel, Bottleneck的个数n， bool(True表示有shortcut，默认为True；反之无)]
            if m is C3:  # BottleneckCSP、C3TR
                args.insert(2, n)
                n = 1   # 恢复默认值1。(C3模块的n，是用来控制其内部bottleneck的个数，而不是C3模块本身的个数。这里将n恢复为1，在下面"得到当前层module"m_的时候，避免计入多个C3模块)
        elif m is Concat:   # Concat层，将f中所有的输出累加，得到这层的输出channel
            c2 = sum([ch[x] for x in f])
        elif m is Detect:   # Detect层
            args.append([ch[x] for x in f])     # 在args中加入三个Detect层的输入channel，128/256/512
        else:
            # Upsample
            c2 = ch[f]  # args不变

        # m_: 得到当前层module   如果n>1就创建多个m(当前层结构)，如果n=1就创建一个m
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')   # module type   'modules.common.Focus'
        np = sum([x.numel() for x in m_.parameters()])  # number params  计算这一层的参数量   # numel() 返回元素个数
        m_.i, m_.f, m_.type, m_.np = i, f, t, np    # attach index索引, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print

        # append to savelist  把所有层结构中from不是-1的值记下 [6, 4, 14, 10, 17, 20, 23]
        # 都来自head部分。Concat[-1, 6]、Concat[-1, 4]、Concat[-1, 14]、Concat[-1, 10]、Detect[17, 20, 23]。取其中的6/4/14/10/17、20、23，代表着yolov5s网络的第6/4/14/10/17、20、23层
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

        # 将当前层结构module加入layers中
        layers.append(m_)

        if i == 0:
            ch = []     # 去除输入channel [3]

        # 把当前层的输出channel数加入ch
        # todo (至于最终Detect层的输出channel，因Detect层没有c2的处理逻辑，个人感觉这行代码对Detect层是无意义的)
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)
