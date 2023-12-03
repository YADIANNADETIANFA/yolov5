# Model validation metrics

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from . import general       # 使用相对导入，导入同一包内的general模块


def fitness(x):
    """
    通过指标加权的形式返回适应度(最终mAP)，在train.py中使用
    Model fitness as a weighted combination of metrics
    判断模型好坏的指标不是 mAP@0.5，也不是 mAP@0.5:0.95，而是[P, R, mAP@0.5, mAP@0.5:0.95] 4者的加权
    一般w=[0, 0, 0.1, 0.9]，即最终的mAP=0.1mAP@0.5 + 0.9mAP@0.5:0.95
    """
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """
    用在test.py中，计算mAP
    计算每一个类的AP指标(average precision)，绘制 P-R 曲线
    mAP基本概念: https://www.bilibili.com/video/BV1ez4y1X7g2
                https://www.bilibili.com/video/BV1ez4y1X7g2/?vd_source=9c2b9b14820d6f6ec6ccc022af406252
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    :param tp: (pred_sum, 10)=(476459, 10) 整个数据集的所有图片的所有预测框，在每一个iou条件下(0.5~0.95)，是否是TP。注意，各元素为bool值。True: 1  False: 0
    :param conf: (img_sum)=(476459) 整个数据集的所有图片的所有预测框的conf
    :param pred_cls: (img_sum)=(476459) 整个数据集的所有图片的所有预测框的类别
          这里的tp、conf、pred_cls是一一对应的
    :param target_cls: (gt_num)=(25022) 整个数据集的所有图片的所有gt框的class
    :param plot: bool
    :param save_dir: runs\train\exp
    :param names: dict{key(class_index): value(class_name)} 数据集所有类别的index和对应类名

    :return p[:, i]: (nc=2) iou阈值为0.5，最大平均f1时，每个类别的precision
    :return r[:, i]: (nc=2) iou阈值为0.5，最大平均f1时，每个类别的recall
    :return ap: (2, 10) 每个类别在10个iou阈值下的AP
    :return f1[:, i]: (nc=2) iou阈值为0.5，最大平均f1时每个类别的f1
    :return unique_classes.astype('int32'): (nc=2) 返回数据集中所有的类别index
    """

    # 计算mAP
    # np.argsort(conf): 不会改变conf的顺序，只是返回一个下标数组i，使得conf[i]正序排序；如果想要降序排序，可使用np.argsort(-conf)
    # 将tp按照conf降序排列
    i = np.argsort(-conf)
    # 得到重新排序后对应的tp、conf、pre_cls
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes 对类别去重，因为计算ap是对每类进行
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections     数据集类别数

    # Create Precision-Recall curve and compute AP
    # px: [0, 1] 中间间隔1000个点，x坐标
    # py: y坐标[] 用于绘制IOU=0.5时的PR曲线
    px, py = np.linspace(0, 1, 1000), []  # for plotting

    # 初始化，准备计算AP P R     ap=(nc, 10)     p=(nc, 1000)    r=(nc, 1000)
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):  # unique_classes: 所有gt中不重复的class
        # 处理所有c类别的预测框
        i = pred_cls == c
        # n_l: gt框中c类别框的数量 = TP+FN  254
        n_l = (target_cls == c).sum()  # number of labels
        # n_p: 预测框中c类别的框数量  695
        n_p = i.sum()  # number of predictions

        # 如果没有预测到，或者ground truth没有标注，则略过类别c
        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FP(False Positive) and TP(True Positive)     FP + TP = all_detections
            # tp[i] 所有tp中属于c类的预测框。各元素为bool值，True: 1  False: 0
            # a.cumsum(0)   会按照对象进行累加操作
            #   参考:  https://blog.csdn.net/banana1006034246/article/details/78841461

            # TP计算公式: {Conf > P_threshold, 且 Iou > Iou_threshold}
            # 对于“某一个Iou”，即Iou_threshold固定
            # 对于“确定该预测框成功预测了c类”，即满足Conf > P_threshold
            # 各个元素，对于True: 1，即Iou > Iou_threshold，则TP累加一分
            # 各个元素，对于False: 0，即Iou < Iou_threshold，TP不累加
            # 累加所有预测框后，即tpc的最后一行，为整个数据集旗下c类别的不同iou的TP分值
            tpc = tp[i].cumsum(0)

            # FP计算公式: {Conf > P_threshold, 且 Iou < Iou_threshold}
            # 对于“某一个Iou”，即Iou_threshold固定
            # 对于“确定该预测框成功预测了c类”，即满足Conf > P_threshold
            # 1-tp[i]: (1-True=0);(1-False=1)
            # 各个元素，对于1-tp[i]: 1，即tp[i]为False，即Iou < Iou_threshold，则FP累加一分
            # 各个元素，对于1-tp[i]: 0，即tp[i]为True，即Iou > Iou_threshold，FP不累加
            # 累加所有预测框后，即fpc的最后一行，为整个数据集旗下c类别的不同iou的FP分值
            fpc = (1 - tp[i]).cumsum(0)     # fp[i] = 1 - tp[i]

            # Recall=TP/(TP+FN)
            # n_l=TP+FN=num_gt: gt框中c类别框的数量
            # 计算各iou阈值下的召回率
            recall = tpc / (n_l + 1e-16)

            # res = numpy.interp(x, xp, fp, left=None, right=None, period=None)，一维线性插值
            # x: 需要进行插值计算的x坐标序列
            # xp: 数据点的x坐标序列
            # fp: 数据点的y坐标序列，与`xp`中的坐标一一对应
            # 其中，xp需保证严格升序或严格降序(一般都是升序)；x可以是乱序无要求；fp与xp对应即可，无顺序要求。
            # 返回值res: x插值计算后，对应的y坐标序列
            #
            # 举例
            # conf = np.array([0.3, 0.5, 0.9, 0.7, 0.1])    # 乱序
            # neg_conf = -conf
            # i = np.argsort(neg_conf)
            # new_conf = conf[i]  # [0.9, 0.7, 0.5, 0.3, 0.1]  降序
            #
            # recall = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # 与new_conf对应的y坐标
            #
            # new_x = np.array([0.6, 0.4, 0.2, 0.8])    # 待插值x序列，乱序
            # res = np.interp(-new_x, -new_conf, recall)    # -new_x仍旧乱序；-new_conf为升序；recall是与new_conf(或-new_conf)相对应的y值
            # print(res)    # [0.4, 0.6, 0.8, 0.2]，可以自己画一下图，是正确的
            #
            # 最终使用`-new_conf`，即[-0.9, -0.7, -0.5, -0.3, -0.1]，是为了能够在计算P-R曲线或其他与置信度相关的度量时，可以优先关注和处理高置信度的预测框
            #       这对评估模型的性能以及决定如何设置置信度阈值非常重要。
            #
            # ci类别，横坐标为conf，纵坐标为recall[:, 0](iou阈值为0.5)，[0:1:1000]的插值结果，r=(nc, 1000)
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)      # 用于绘制R-Confidence(R_curve.png)

            # Precision=TP/(TP+FP)
            # 计算各iou阈值下的准确率
            precision = tpc / (tpc + fpc)  # precision curve    用于计算mAP

            # ci类别，横坐标为conf，纵坐标为precision[:, 0](iou阈值为0.5)，[0:1:1000]的插值结果，p=(nc, 1000)
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # 用于绘制P-Confidence(P_curve.png)

            # tp[i]是i类预测框按conf降序排序的。即从上到下，各预测框的conf越来越小。
            # 在累加各预测框的过程中，从上到下:
            #       tpc从小变大(但增大的速度越来越慢，因为越到下面，预测框的conf就越小，为True的概率就越小)
            #       fpc从小变大(但增大的速度越来越快，因为越到下面，预测框的conf就越小，为False的概率就越大)
            # 因此，根据recall与precision的计算公式，在累加各预测框的过程中，从上到下:
            #       recall越来越大
            #       precision越来越小
            # 这也是符合P-R曲线特点的

            # P-R曲线，横坐标为Recall值，纵坐标为Precision值
            # 详见 https://www.bilibili.com/video/BV1ez4y1X7g2/?vd_source=9c2b9b14820d6f6ec6ccc022af406252

            # AP from recall-precision curve
            for j in range(tp.shape[1]):    # tp.shape[1]: 10个iou等级
                # 这里执行10次，计算ci这个类别在各个iou阈值下的ap
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # py: 后续用于绘制每一个类别在IOU=0.5时的PR曲线

    # Compute F1 (harmonic mean of precision and recall 准确率和召回率的调和平均值)
    # 计算F1分数，P和R的调和平均值，综合评价指标
    # 我们希望的是P和R两个越大越好, 但是P和R常常是两个冲突的变量, 经常是P越大R越小, 或者R越大P越小 所以我们引入F1综合指标
    # 不同任务的重点不一样, 有些任务希望P越大越好, 有些任务希望R越大越好, 有些任务希望两者都大, 这时候就看F1这个综合指标了
    # 返回所有类别, 横坐标为conf(值为px=[0:1:1000] 0~1 1000个点)对应的f1值  f1=(nc, 1000)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)   # 画pr曲线
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')  # 画F1_conf曲线
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')     # 画P_conf曲线
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')    # 画R_conf曲线

    # f1=(nc, 1000)     f1.mean(0) 求出所有类别在x轴每个conf点上的平均f1
    # .argmax(): 最大平均f1所对应的conf点的index
    i = f1.mean(0).argmax()  # max F1 index

    # p[:, i]: iou=0.5时，所有类别最大平均f1时，各个类别的precision
    # r[:, i]: iou=0.5时，所有类别最大平均f1时，各个类别的recall
    # ap: (nc=2, 10) 各个类别分别在10个iou下的AP
    # f1[:, i]: iou=0.5时，所有类别最大平均f1时，各类别分别的f1值
    # unique_classes.astype('int32'): (nc=2) 数据集中所有的类别index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """
    用于ap_per_class中
    计算某个类别在某个iou阈值下的ap
    Compute the average precision, given the recall and precision curves
    :param recall: 某个类别，在某个iou阈值下，上面所有预测框的recall
                    (每个预测框的recall都是截至到这个预测框为止的总recall，详见recall处的说明)
    :param precision: 某个类别，在某个iou阈值下，上面所有预测框的precision
                    (每个预测框的precision都是截至到这个预测框为止的总precision，详见precision处的说明)

    :return ap: Average precision，某个类别在某个iou阈值下的ap
    :return mpre: 传入的precision，前面额外拼一个元素"1.0"，后面额外拼一个元素"0.0"
    :return mrec: 传入的recall，前面额外拼一个元素"0.0"，后面额外拼一个元素"recall[-1] + 0.01"
    """

    # 在开头和末尾添加保护值，防止全零的情况出现     Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))      # 从0到1
    mpre = np.concatenate(([1.], precision, [0.]))      # 从1到0

    # Compute the precision envelope  np.flip翻转顺序
    # np.flip(mpre): mpre元素顺序翻转，从0到1
    # np.maximum.accumulate(np.flip(mpre)): 对给定的一维数组，返回每个位置到当前为止的最大值。[2,1,5,3,7,4,6] -> [2,2,5,5,7,7,7]。即让mpre严格单调，从小到大，从0到1
    # np.flip(np.maximum.accumulate(np.flip(mpre))): 再次翻转，此时，从大到小，从1到0

    # mpre原本就是从1到0，从大到小。这里的操作只是为了让mpre保证严格单调递减而已，并不是为了刻意去改变mpre的值。
    # 这样是为了更好地计算mAP，因为如果一直起起伏伏不严格单调，不太好算(x间隔很小就是一个矩形)。而且这样做误差也不会很大，两个之间的数都是间隔很小的。
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':  # 用一些典型的间断点来计算AP
        x = np.linspace(0, 1, 101)  # 101-point interp [0, 0.01, ..., 1]
        # np.trapz(list, list) 计算两个list对应点与点之间所成图形的面积，以定积分形式估算ap，第一个参数是y，第二个参数是x。(使用复合梯形规则沿给定的轴进行积分)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'   # 采用连续的方法计算AP
        # 通过错位的方式，判断哪个点当前位置到下一个位置值发生改变，并通过 != 判断，返回一个布尔数组
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        # 值改变了就求出当前矩形的面积，值没变就说明当前矩形和下一个矩形的高相等，所以可以合并计算
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


# True Positive(TP): A correct detection. Detection with IOU >= threshold
# False Positive(FP): A wrong detection. Detection with IOU < threshold
# False Negative(FN): A ground truth not detected
# True Negative(TN): Does not apply. It would represent a corrected misdetection. In the objection task
#       there are many possible bounding boxes that should not be detected within a image. Thus, TN would be
#       all possible bounding boxes that are correctly not detected (so many possible boxes within an image).
#       That's why it is not used by the metrics.
class ConfusionMatrix:
    """
    用在test.py中，计算混淆矩阵
    Updated version of https://github.com/kaanakan/object_detection_confusion_matrix

    混淆矩阵，x横轴(右方向)为gt真实值；y纵轴(上方向)为pred预测值

    混淆矩阵基础知识
    https://blog.csdn.net/weixin_43745234/article/details/121561217

    yolov5混淆矩阵说明
    https://blog.csdn.net/Yonggie/article/details/126892359
    注意：
        1. `background类`就是`background类`，并没有什么`background FN类`和`background FP类`，他只是想说明，最后一行的值是计算的background FN；最后一列的值是计算的background FP
        2. background FP：本来是物体，但被预测成了背景，即`漏检`了非背景物体；
           background FN：本来是背景，但被预测成了物体，即`虚检`了本来没有的物体。
    """
    def __init__(self, nc, conf=0.25, iou_thres=0.45):  # 个人觉得这里iou_thres应该改成0.5(和后面计算mAP对应)
        """
        :param nc: 数据集类别个数
        :param conf: 预测框置信度阈值
        :param iou_thres: iou阈值
        """
        self.matrix = np.zeros((nc + 1, nc + 1))    # +1为背景类(background)
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        :param detections: (N, 6) = (pred_obj_num, x1y1x2y2 + object_conf + cls) = (130, 6)
                            一个batch中一张图的预测信息，其中x1y1x2y2是映射到原图img的
        :param labels: (M, 5) = (gt_num, class + x1y1x2y2) = (1, 5) 其中x1y1x2y2是映射到原图img的
        :return: None, updates confusion matrix accordingly
        """
        # 筛除置信度过低的预测框(和nms差不多)
        detections = detections[detections[:, 4] > self.conf]

        # 所有gt框类别(int)，类别可能会重复
        gt_classes = labels[:, 0].int()
        # 所有pred框类别(int)，类别可能会重复 Positive + Negative
        detection_classes = detections[:, 5].int()

        # 求出所有gt框和所有pred框的iou (17, x1y1x2y2) + (10, x1y1x2y2) => (17, 10)   (i, j): 第i个gt框和第j个pred的iou
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        # iou > self.iou_thres: (17, 10) bool 符合条件True，不符合False
        # x合起来看，就是第x[0]个gt框和第x[1]个pred的iou符合条件
        # 17 * 10个iou，经过iou阈值筛选后只有10个满足iou阈值条件的
        x = torch.where(iou > self.iou_thres)

        if x[0].shape[0]:
            # 存在大于阈值的iou时
            # torch.stack(x, 1): (10, gt_index + pred_index)
            # iou[x[0], x[1]][:, None]: [10, 1]     x[0]和x[1]的iou
            # 1、matches: (10, gt_index + pred_index + iou) = (10, 3)
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # 2、matches按第三列iou从大到小重排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 3、取第二列中各个框首次出现(不同预测的框)的行(即每一种预测的框中iou最大的那个)
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # 4、matches再按第三列iou从大到小重排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 5、取第一列中各个框首次出现(不同gt的框)的行(即每一种gt框中iou最大的那个)
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]   # (9, gt_index + pred_index + iou)
                # 经过这样的处理，最终得到每一种预测框与所有gt框中iou最大的那个(在大于阈值的前提下)
                # 预测框唯一，gt也唯一。这样得到的matches对应的Pred都是正样本Positive
        else:
            matches = np.zeros((0, 3))

        # bool  满足条件的iou是否大于0个
        n = matches.shape[0] > 0
        # matches: (9, gt_index + pred_index + iou) => (gt_index + pred_index + iou, 9)
        # m0: (1, 9) 满足条件(正样本)的gt框index(不重复)    m1: (1, 9) 满足条件(正样本)的pred框index(不重复)
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                # 如果sum(j)=1，说明gt[i]这个真实框被某个预测框检测到了，但是detection_classes[m1[j]]并不一定等于gc，所以此时可能是TP或者FP
                # m1[j]: gt框index=i时，满足条件的pred框index，   detection_classes[m1[j]]: pred_class_index
                # gc: gt_class_index
                # 即：matrix[pred_class_index, gt_class_index] += 1
                # 某个gt被检测到了，但pred的分类结果可能是对的也可能是错的。(取决于gc和detection_classes[m1[j]]是否一致)
                self.matrix[detection_classes[m1[j]], gc] += 1
            else:
                # 如果sum(j)=0，说明gt[i]这个真实框没有被任何预测框检测到，也就是说这个真实框被检测成了背景框
                # 所以对应的混淆矩阵 [背景类, gc] += 1。(最后一个类别为背景类)
                # 某个gt没被检测到，被pred为background了
                self.matrix[self.nc, gc] += 1

        if n:
            for i, dc in enumerate(detection_classes):
                # 迭代每一个预测框
                if not any(m1 == i):
                    # 首先，这是一个预测框，是预测框就有预测类别。
                    # 但是经上面的条件判决后发现，该次迭代的预测框不满足条件，即该次迭代的预测框是一个负样本。
                    # 负样本，就是背景，也就是理论上(真实情况上)来说，这个预测框实际上应该是个背景，但是我们却用它做了某个类别的预测。
                    self.matrix[dc, self.nc] += 1

    # 返回这个混淆矩阵
    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        """
        :param save_dir: runs/train/expn 混淆矩阵保存地址
        :param names: 数据集的所有类别名
        :return: None
        """
        try:
            import seaborn as sn    # 为matplotlib可视化更好看的一个模块

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize  混淆矩阵归一化    (sum(0)，计算每一列的和)
            array[array < 0.005] = np.nan  # 混淆矩阵中小于0.005的值被认为NaN

            fig = plt.figure(figsize=(12, 9), tight_layout=True)    # 初始画布
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size   设置label的字体大小
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels   绘制混淆矩阵时，是否使用names作为labels

            # 绘制热力图 即混淆矩阵可视化
            # sean.heatmap: 热力图  data: 数据矩阵  annot: 为True时为每个单元格写入数据值 False用颜色深浅表示
            # annot_kws: 格子外框宽度  fmt: 添加注释时要使用的字符串格式代码 cmap: 指色彩颜色的选择
            # square: 是否是正方形  xticklabels、yticklabels: xy标签
            # 不必在意这里的`background FP/FN`，以background类去理解即可
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            # 设置figure的横坐标 纵坐标及保存该图片
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    # print按行输出打印混淆矩阵matrix
    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    """用于ap_per_class函数
    Precision-recall curve  绘制P-R曲线
    :params px: [1000] 横坐标 recall 值为0~1直接取1000个数
    :params py: list{nc} nc个[1000] 所有类别在IOU=0.5,横坐标为px(recall)时的precision
    :params ap: [nc, 10] 所有类别在每个IOU阈值下的平均mAP
    :params save_dir: runs\test\exp54\PR_curve.png  PR曲线存储位置
    :params names: {dict: 2} 数据集所有类别的字典 key:value
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)     # 设置画布
    py = np.stack(py, axis=1)   # [1000, nc]

    # 画出各个类别在Iou=0.5下的P-R曲线
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):    # 如果<21 classes就一个个类画 因为要显示图例就必须一个个画
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:   # 如果>=21 classes 显示图例就会很乱 所以就不显示图例了 可以直接输入数组 x[1000] y[1000, 71]
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    # 画出所有类别在IOU=0.5下的平均PR曲线
    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')     # 设置x轴标签
    ax.set_ylabel('Precision')  # 设置y轴标签
    ax.set_xlim(0, 1)           # x=[0, 1]
    ax.set_ylim(0, 1)           # y=[0, 1]
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")      # 显示图例
    fig.savefig(Path(save_dir), dpi=250)                        # 保存PR_curve.png图片


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    """用于ap_per_class函数
    Metric-Confidence curve 可用于绘制 F1-Confidence/P-Confidence/R-Confidence曲线
    :params px: [0, 1, 1000] 横坐标 0-1 1000个点 conf   [1000]
    :params py: 对每个类，针对横坐标为conf=[0, 1, 1000]，对应的f1/p/r值 纵坐标 [2, 1000]
    :params save_dir: 图片保存地址
    :parmas names: 数据集中各类别名
    :params xlabel: x轴展示的标签
    :params ylabel: y轴展示的标签
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)     # 设置画布

    # 画出所有类别的F1-Confidence/P-Confidence/R-Confidence曲线
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):  # 如果<21 classes就一个个类画 因为要显示图例就必须一个个画
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:   # 如果>=21 classes 显示图例就会很乱 所以就不显示图例了 可以直接输入数组 x[1000] y[1000, 71]
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    # 画出所有类别在每个x点(conf)对应的均值F1-Confidence/P-Confidence/R-Confidence曲线
    y = py.mean(0)  # [1000] 所有类别在每个x点(conf)的平均值
    # max()返回序列中的最大元素值；argmax()返回数组中最大元素所在位置的索引
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)  # 设置x轴标签
    ax.set_ylabel(ylabel)  # 设置y轴标签
    ax.set_xlim(0, 1)  # x=[0, 1]
    ax.set_ylim(0, 1)  # y=[0, 1]
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # 显示图例
    fig.savefig(Path(save_dir), dpi=250)  # 保存png图片
