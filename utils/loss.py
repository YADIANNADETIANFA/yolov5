# Loss functions

import torch
import torch.nn as nn
from utils.general import bbox_iou


def smooth_BCE(eps=0.1):
    """
    用在ComputeLoss类中
    标签平滑操作 [1, 0] -> [0.95, 0.05]
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    :param eps: 平滑参数
    :return positive, negative label smoothing BCE targets: 两个值分别代表正样本和负样本的标签取值
            原先的正样本=1，负样本=0，改为 正样本=1.0-0.5*eps  负样本=0.5*eps
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria   定义分类损失和置信度损失
        # 衡量目标和输出之间的二进制交叉熵
        # https://blog.csdn.net/qq_38253797/article/details/116225218
        # https://blog.csdn.net/qq_22210253/article/details/85222093
        # https://www.jianshu.com/p/0062d04a2782
        # pos_weight 样本不均衡处理: https://blog.csdn.net/qq_37451333/article/details/105644605
        #   即若有需要，可通过pos_weight对正负样本的loss进行加权处理，将正样本的loss权重放大n倍，缓解样本不均衡的问题
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))    # 分类损失    # 暂未进行样本均衡处理
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))    # 置信度损失   # 暂未进行样本均衡处理

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # self.cp代表正标签值(1.0)     self.cn代表负样本标签值(0.0)
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets      暂未进行标签平滑

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma   0.0   暂未用FocalLoss替代BCEcls与BCEobj
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # det: 返回的是模型的检测头 Detector 3个 分别对应产生三个输出feature map
        det = model.model[-1]  # Detect() module

        # balance用来设置三个feature map对应输出的置信度损失系数(平衡三个feature map的置信度损失)
        # 从左到右分别对应大feature map(检测小目标)到小feature map(检测大目标)
        # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
        # self.balance = [4.0, 1.0, 0.4]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])

        # 这个参数可用来自动计算更新3个feature map的置信度损失系数self.balance，但我们固定了self.balance的值，不使用这里的自动计算
        self.ssi = 0

        # self.BCEcls: 类别损失函数
        # self.BCEobj: 置信度损失函数
        # self.gr: 计算真实框的置信度标准的iou ratio
        # self.hyp: 超参数
        # self.autobalance: 是否自动更新各feature map的置信度损失平衡系数  默认False
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance

        # na: number of anchors  每个grid_cell的anchor数量 = 3
        # nc: number of classes  数据集的总类别 = 80   我们这里是2
        # nl: number of detection layers   Detect的个数 = 3
        # anchors: [3, 3, 2]  3个feature map 每个feature map上有3个anchor(w,h) 这里的anchor尺寸是相对feature map的
        for k in 'na', 'nc', 'nl', 'anchors':
            # setattr: 给对象self的属性k赋值为getattr(det, k)
            # getattr: 返回det对象的k属性
            # 所以这句话的意思: 将det的k属性赋值给self.k属性 其中k in 'na', 'nc', 'nl', 'anchors'
            setattr(self, k, getattr(det, k))

    # 相当于forward函数，在这个函数中进行损失函数的前向传播
    def __call__(self, p, targets):  # predictions, targets
        """
        :param p: 预测框，三个检测头Detector返回的三个yolo层的输出。是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值，后面肯定要进行正样本筛选。
                    (bs, anchor_num, grid_h, grid_w, xywh+conf+2个class的预测概率)
                    如: (32, 3, 80, 80, 7) (32, 3, 40, 40, 7) (32, 3, 20, 20, 7)
        :param targets: 数据增强后的真实框。 (n, 6)   即 (num_target(整个batch中所有gt框的个数), img_index(该gt框归属于该batch中的哪一张img) + class_index(0:hat, 1:person) + xywh(normalized))
        :return loss * bs: 整个batch的总损失，进行反向传播  (没太搞明白这里，为什么`loss * bs`是整个batch的总损失...)
        :return torch.cat((lbox, lobj, lcls, loss)).detach(): 回归损失、置信度损失、分类损失和总损失，这个参数只用来可视化
        """
        device = targets.device
        # 初始化lcls, lbox, lobj三种损失值 tensor([0.])
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # 遍历feature map(3)次，将每次的结果append起来。每次的结果都是当前feature map筛选出的anchor正样本
        # 对于某一个target(gt)框，其可以由当前feature map下的、某个grid cell的，某个anchor的"类别"进行预测，则这个feature map下的这个grid cell下的这个anchor"类别"，就是一个正样本。
        #       (注意，一个target或许可以用该grid cell下的多个anchor"类别"进行预测(单个feature map下，每个grid cell有3种anchor)，它们都有可能是正样本)
        #       个人理解：所谓正样本，指的不是target(gt)框，也不是yolov5网络输出的anchor预测框。
        #               正样本指的是各个feature map下，各个grid cell中，3种"初始"anchor类别，谁的宽高比与gt符合条件，那么这个或这几个"初始"anchor类别，就是当前这个grid cell下，这个gt框的正样本。
        #               而且你也可以看到，在self.build_targets()挑选正样本的过程中，只涉及了"初始"anchor的宽高，并不涉及anchor的坐标。挑选正样本是指挑选各个grid cell下，将要对gt框进行预测的anchor类别，正样本指的不是pred预测框。
        #               后续，挑选pred预测框中的正样本框，也就是"ps = pi[b, a, gj, gi]"中的ps，计算loss、反向传播，让这些正样本框越来越接近gt框的位置和大小。
        # tcls: 表示这个target所属的class index
        # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
        # indices: b: 表示这个target属于的image index
        #          a: 表示这个target使用的anchor index(0,1,2)     个人理解，它们就是真正的正样本
        #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失) gj表示这个网格的左上角y坐标
        #          gi: 表示这个网格的左上角x坐标
        # anchors: 表示这个target所使用anchor的尺度(相对于这个feature map)    注意可能一个target会使用大小不同anchor进行计算    个人理解，就是正样本的尺度
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        # 依次遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj    初始化target置信度(先全是负样本，后面再筛选正样本赋值)

            n = b.shape[0]  # 正样本数量
            if n:
                # pi: 当前feature map下，yolov5输出的所有预测框值     e.g (32, 3, 80, 80, 7)，代表一个batch的32张img，3种不同形状的anchor，80*80的grid数量，7为详细数据(xywh+conf+2个class的预测概率)
                # b: e.g shape(3239)，value:0~31  每一个target属于该batch中哪一张img                  与pi的32对应
                # a: e.g shape(3239)，value:0,1,2   grid cell下，每一个target的正样本是3种anchor中的哪一个                 与pi的3对应
                # gi,gj: e.g shape(), value:0~79   每一个target所对应的grid cell的左上角xy坐标       与pi的80*80对应
                # pi是当前feature map下的所有预测框；ps是pi的正样本筛选，即当前feature map下的所有正样本的预测框
                # ps = pi[b, a, gj, gi]     ps: 第b张img的、第a种anchor的、第gi,gj grid cell网格的，yolov5网络输出预测框值。即当前feature map下的所有正样本预测框。
                # 用这些正样本预测框与对应的target(gt)真实框进行loss计算
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression loss   只计算所有正样本的回归损失       ps[:, :2], 7中的前两项，即xy，计算回归损失
                # 新的公式: pxy = [-0.5 + cx, 1.5 + cx]     pwh = [0, 4pw]  这个区域内都是正样本
                # Get more positive samples, accelerate convergence and be more stable(获取更多正样本，加速收敛，更稳定)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5    # 一个归一化操作，和论文里不同
                # https://github.com/ultralytics/yolov3/issues/16
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]      # 和论文里不同，这里是作者自己提出的公式
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # tbox[i]中的xy，是这个正样本对当前grid_cell左上角的偏移量[0, 1]。而pbox.T是一个归一化的值
                # 就是要用这种方式训练，传回loss，修改梯度，让pbox越来越接近tbox(偏移量)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # iou.detach() 不会更新iou梯度，iou并不是反向传播的参数，所以不需要反向传播梯度信息
                # .clamp(0)必须大于等于0
                # 预测信息有置信度，但是真实框信息是没有置信度的，所以需要我们人为的给一个标准置信度
                # self.gr是iou ratio [0, 1]  self.gr越大置信度越接近iou  self.gr越小置信度越接近1(人为加大训练难度)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification loss   只计算所有正样本的分类损失
                if self.nc > 1:  # cls loss (only if multiple classes. 多个类别才会有分类损失，hat、person就是多类别)
                    # full_like     https://blog.csdn.net/Fluid_ray/article/details/109855155
                    # target 原本负样本是0， 这里使用smooth label，就是cn
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp  # 筛选到的正样本对应位置值是cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            # Objectness loss 置信度损失是用所有样本(正样本+负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            # 每个feature map的置信度损失权重不同，要乘以相应的权重系数self.balance[i]
            # 一般来说，检测小物体的难度大一些，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:    # 我们这里是False
                # 自动更新各个feature map的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 根据超参中的损失权重参数，对各个损失进行平衡，防止总损失被某个损失所左右
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls   # 平均每张图片的总损失  (没太搞明白这里，为什么是"平均每张图片的总损失"...)

        # loss * bs: 整个batch的总损失  (没太搞明白这里，为什么`loss * bs`是整个batch的总损失...)
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        """
        所有target(gt)框，筛选相应的anchor正样本
        Build targets for compute_loss()

        参考链接
        https://blog.csdn.net/weixin_46142822/article/details/123820031

        https://www.cnblogs.com/AIBigTruth/p/16876004.html
        什么是正负样本?
            正负样本是在训练过程中计算损失用的，而在预测过程和验证过程是没有这个概念的。
            正样本并不是手动标注的gt框。
            正负样本都是针对于算法经过处理生成的框而言，而非原始的gt框数据。
            正样本是用来使预测结果更靠近真实值的，负样本用于训练模型来识别什么不是目标(有助于模型识别哪些对象不应该被检测)。

        个人理解：
        1、target(gt)框的xywh，是“人工标注框”的坐标和宽高。注意，“人工标注框”不是“anchor锚框”，前者是真实标签数据，和anchor没什么关系。
        2、预测值p只用来获取feature map的尺度数据，不参与正样本的筛选工作。
        3、筛选正样本 =
                ”单个feature map下，单个grid cell下(该target中心点坐标所在的grid cell)，通过3个anchor与target的宽高比(只关心target人工标注框与anchor锚框的宽高比，不关心位置坐标)，筛选出该层feature map下、该grid cell下的正样本anchor(正样本anchor数<=3)。“
                "单个feature map下，根据target的中心点坐标，再选出两个近一些的grid cell。还是用上面的方法，筛选出这3个(2+1)grid cell下的正样本anchor(正样本anchor数<=3*3)"
                "每一层feature map都这样做，筛选出所有feature map的正样本anchor(正样本anchor数<=3*3*3)"
        4、什么是p? p是包含了所有feature map、所有grid cell，所有anchor的预测值。即p是所有anchor的预测值。
            对任意一个anchor，若其属于正样本，就会用正样本的方式计算该anchor预测值与target的loss；
                            若其属于负样本，就会用负样本的方式计算该anchor预测值与target的loss。
            即对正负样本进行不同的loss计算。
        5、target最初来源于 VOCdevkit/VOC2007/YOLOLabels，其xywh可参考voc_convert_yolo.py的注释

        :param p: p[i]的作用只是得到每个feature map的shape
        :param targets: 数据增强后的gt真实框 (n, 6)      (num_target(整个batch中所有gt框的个数), image_index(该gt框归属于该batch中的那一张img)+class(0:hat, 1:person)+xywh(normalized))
        :return tcls: 表示这个target所属的class index
                tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                indices: b: 表示这个target属于的image index
                         a: 表示这个target使用的anchor index(0,1,2)     个人理解，它们就是真正的正样本
                         gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失) gj表示这个网格的左上角y坐标
                         gi: 表示这个网格的左上角x坐标
                anch: 表示这个target所使用anchor的尺度(相对于这个feature map)    注意可能一个target会使用大小不同anchor进行计算    个人理解，就是正样本的尺度
        """
        na, nt = self.na, targets.shape[0]  # number of anchors 3, targets n
        tcls, tbox, indices, anch = [], [], [], []  # 初始化

        # gain是为了后面将targets(shape:(na, nt, 7))中的归一化了的xywh映射到相对feature map尺度上
        # 7: image_index+class+xywh+anchor_index
        # 从后续对gain的赋值也可以看到，仅对gain[2:6]进行了赋值，即xywh部分。而image_index、class、anchor_index都始终保持为1
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain

        # 准备在下面，为每个target添加上anchor的索引。(在某一层feature map上。)
        # (1, 3) -> (3, 1) -> (3, nt)=(na, nt)  三行 第一行nt个0 第二行nt个1 第三行nt个2
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # (3, nt, 6) (3, nt, 1) -> (3, nt, 7)    7: (image_index+class+xywh+anchor_index)
        # 每个feature map的每个grid cell，都有三个anchor。(这三种anchor隶属于同一feature map，它们大小相近，但形状稍有不同)
        # 将target复制三份，分别对应单层feature map的三个anchor。   第一份target，所有anchor编号都是0；第二份target，所有anchor编号都是1；第三份target，所有anchor编号都是2
        # 对于任意一层的feature map，我们认为所有的target都由该层的feature map的三种anchor进行检测(复制三份)(这三种anchor隶属于同一feature map，它们大小相近，但形状稍有不同)。
        # 稍后进行筛选。将ai加进去，为每个target添加上anchor的索引。
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        # 这两个变量是用来扩展正样本的，因为预测框预测到target有可能不止当前的格子预测到了
        # 可能周围的格子也预测到了高质量的样本，我们也要把这部分的预测信息加入正样本中
        g = 0.5  # bias 中心偏置，用来衡量target中心点离哪个格子更近
        # 以自身 + 周围左上右下四个网格 = 5个网格，用来计算offsets。**** 注意最下面的 *g ****
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 遍历三个feature map，筛选target(gt)框的anchor正样本
        for i in range(self.nl):    # self.nl: number of detection layers   Detect的个数 = 3
            # anchors: 当前feature map对应的三个anchor尺寸(相对feature map) [3, 2]
            anchors, shape = self.anchors[i], p[i].shape

            # 保存当前feature map的尺度变换。     gain[2:6]=gain[xywh]
            # (1, 1, 1, 1, 1, 1, 1) -> (1, 1, 80/40/20, 80/40/20, 80/40/20, 80/40/20, 1) = image_index+class+xywh+anchor_index
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # feature map的shape

            # t = (3, nt, 7) 将target中的xywh缩放到相对当前feature map的坐标尺度
            #     (3, nt, image_index+class+xywh+anchor_index)
            t = targets * gain

            if nt:  # target确实存在，开始进行匹配
                # None与np.newaxis效果一样，None是np.newaxis的别名
                # https://blog.csdn.net/qq1483661204/article/details/73135766
                # https://blog.csdn.net/qq_36490364/article/details/83594271
                # t[:, :, 4:6]=(na, nt, wh)=(3, nt, 2)
                # anchors[:, None]: (3, 1, 2)，广播后将与t[:,:, 4:6]对应
                # r=(na, nt, 2)=(3, nt, 2)
                # 所有的target(gt)与当前feature map的三个anchor的宽高比(gt_w/anchor_w, gt_h/anchor_h)
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio

                # 筛选条件: target(gt)与anchor的宽比或高比超过一定的阈值，就当做负样本
                # torch.max(r, 1./r)=[3, nt, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(dim=2): 在第三个维度(0,1,2)上进行max选取。返回宽比、高比中较大的值和它的索引。  (torch.max(r, 1. / r).max(dim=2)[0]返回较大的值; torch.max(r, 1. / r).max(dim=2)[1]返回较大的值的索引)
                # j: [3, nt] False: 当前anchor是当前target(gt)的负样本   True: 当前anchor是当前target(gt)的正样本
                j = torch.max(r, 1. / r).max(dim=2)[0] < self.hyp['anchor_t']  # compare
                # yolov3 v4的筛选方法: target(gt)与anchor的wh_iou超过一定的阈值就是正样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                # 根据筛选条件j，过滤负样本，得到所有target(gt)框的anchor正样本
                # t: (3, nt, 7) -> (1068, 7)  (num_Positive_sample, image_index+class+xywh+anchor_index)
                t = t[j]  # filter

                # Offsets 筛选当前格子周围格子，找到离target中心最近的两个格子。可能周围的格子也预测到了高质量的样本，我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外，还有2个格子对目标进行检测(计算损失)，也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个，再从当前格子的上下左右四个格子中选择2个，用这三个格子去预测这个目标(计算损失)
                # feature map整图的原点在左上角，向右为x轴正坐标，向下为y轴正坐标
                gxy = t[:, 2:4]  # gxy: target中心点坐标xy(相对于feature map整图左上角)
                gxi = gain[[2, 3]] - gxy  # gxi: target中心点，相对于feature map整图右下角的坐标  gain[[2, 3]]为当前feature map整图的右下角
                # 筛选中心坐标，距离当前grid_cell的左、上方偏移小于g=0.5，且中心坐标必须大于1(坐标不能在边上，此时就没有4个格子了)
                # j: [1068] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [1068] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                # (gxy > 1.)，确保当前中心格子不贴在feature map整图的左边缘和上边缘，这样才能取到当前格子的左边格和上边格
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # l: [1068] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [1068] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                # (gxi > 1.)，确保当前中心格子不贴在feature map整图的右边缘和下边缘，这样才能取到当前格子的右边格和下边格
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T

                # torch.ones_like(): 根据给定的张量，生成与其形状相同的全1张量
                # torch.stack(): 张量拼接并升维。(torch.cat(): 张量拼接但不升维)
                # torch.one_like(j): 当前的中心格子，不需要筛选，全是True；  j,k,l,m: 左上右下格子的筛选结果
                # j: (5, 1068)
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本，格子数<=3*1068，都不在边上时等号成立
                # t: (1068, 7) -> 复制5份: (5, 1068, 7) 分别对应当前格子和左上右下格子5个格子
                # j: (5, 1068) & t: (5, 1068, 7) => t: (3188, 7) 理论上是小于等于3倍的1068，当且仅当没有边界的格子时，等号成立
                t = t.repeat((5, 1, 1))[j]

                # torch.zeros_like(gxy)[None]: (1, 1068, 2)  off[:, None]: (5, 1, 2) => (5, 1068, 2)
                # 中心网格(target中心点所在grid cell)与上下左右共5个网格。
                # target中心点所在的网格，相对这五个网格的偏移量
                # j是正样本筛选条件。j筛选后: (3188, 2) 得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下框）的偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # 这里，t是所有的正样本
            b, c = t[:, :2].long().T  # 所有target的image_index、class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()    # 用来预测真实框的所有网格(必包括中心网格，可能包含上下左右网格)，所在的左上角坐标（有左上右下的网格）
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # b: image index    a: anchor index     gj: 网格的左上角y坐标       gi: 网格的左上角x坐标
            indices.append((b, a, gj.clamp_(0, (gain[3] - 1).type(torch.int64)), gi.clamp_(0, (gain[2] - 1).type(torch.int64))))  # image, anchor, grid indices
            # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors  对应的所有anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
