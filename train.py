import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_schedule
import torch.utils.data
import wandb
import yaml
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import test                 # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, check_img_size, set_logging, one_cycle, colorstr
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import ModelEMA, intersect_dicts
from utils.wandb_logging.wandb_utils import WandbLogger

logger = logging.getLogger(__name__)

# todo 改成你的 WANDB_API_KEY
os.environ["WANDB_API_KEY"] = "your_wandb_api_key"
wandb.login()


def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    save_dir = Path(opt.save_dir)   # 保存训练结果的目录  如runs/train/exp18
    epochs = opt.epochs
    batch_size = opt.batch_size
    total_batch_size = opt.total_batch_size
    weights = opt.weights   # './weights/yolov5s.pt'
    rank = opt.global_rank  # -1

    # Directories 目录
    wdir = save_dir / 'weights'     # 保存权重路径 如runs/train/exp/weights
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'         # runs/train/exp/weights/last.pt
    best = wdir / 'best.pt'         # runs/train/exp/weights/best.pt

    # 记录训练结果和验证结果，这里先记录表头信息。(原始代码中是没有这个表头信息的，这里是我自己加的。这会影响到最终results.png绘制处的代码(plots.py  plot_results()函数)，我们也一并修改好了)
    #   train_epoch/total_epochs: 当前epoch/total_epochs
    #   train_gpu_mem_reserved: 显存
    #   train_box_loss: 训练集回归损失
    #   train_obj_loss: 训练集置信度损失
    #   train_cls_loss: 训练集分类损失
    #   train_total_loss: 训练集总损失
    #   train_labels_num: 训练集当前batch的target的数量
    #   train_img_size: 训练集当前batch的图片size
    #   val_precision: 验证集iou=0.5时，所有类别最大平均f1时，所有类别的平均precision
    #   val_Recall: 验证集iou=0.5时，所有类别最大平均f1时，所有类别的平均recall
    #   val_mAP@.5: 验证集所有类别的平均AP@0.5，即map@0.5
    #   val_mAP@.5:.95: 验证集所有类别的平均AP@0.5:0.95，即map@0.5:0.95
    #   val_box_loss: 验证集回归损失
    #   val_obj_loss: 验证集置信度损失
    #   val_cls_loss: 验证集分类损失
    results_file = save_dir / 'results.txt'  # runs/train/exp/results.txt
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(('%25s' * (8 + 7)) % ('train_epoch/total_epochs', 'train_gpu_mem_reserved', 'train_box_loss', 'train_obj_loss',
                                      'train_cls_loss', 'train_total_loss', 'train_labels_num', 'train_img_size', 'val_precision',
                                      'val_Recall', 'val_mAP@.5', 'val_mAP@.5:.95', 'val_box_loss', 'val_obj_loss', 'val_cls_loss') + '\n')

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)      # 记录训练的超参数
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)    # 记录训练的命令行设置参数

    plots = True
    cuda = True
    # 设置一系列的随机数种子
    init_seeds(2 + rank)

    with open(opt.data) as f:   # ./data/hat.yaml
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    is_coco = False

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    opt.hyp = hyp   # add hyperparameters
    run_id = torch.load(weights).get('wandb_id')    # None
    wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
    loggers['wandb'] = wandb_logger.wandb
    data_dict = wandb_logger.data_dict
    if wandb_logger.wandb:
        # WandbLogger might update weights，epochs. if resuming.    我们这里resume=False，所以不会更新
        weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp


    nc = int(data_dict['nc'])  # number of classes  2   数据集有多少种类别
    names = data_dict['names']  # class names   ['hat', 'person']  数据集所有类别的名字
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    ckpt = torch.load(weights, map_location=device)  # load checkpoint      预训练权重参数
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 输入ch三通道      分类数nc=2
    exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys ['anchor']
    state_dict = ckpt['model'].float().state_dict()
    # 判断state_dict与model.state_dict()哪些参数是相同的（毕竟nc分类数是自定义的，预训练权重中的nc是80，而此时我们nc的分类数是2）
    # intersect 相交      筛选字典中的键值对，把exclude删除
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
    model.load_state_dict(state_dict, strict=False)  # 载入模型权重
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report

    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze    冻结权重层
    # 不对其参数进行训练。我们未做冻结操作。(训练全部层参数，可以得到更好的性能，当然也会更慢)
    freeze = []
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers    如果想冻结某些层，则将这些层的requires_grad设为False，即不计算Tensor的梯度
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    # nbs为模拟batch_size;
    # 我们设置opt.batch_size为32,nbs为64。此时，模型会在梯度累积64/32=2(accumulate)次之后，对模型进行一次更新；
    # 即变相扩大了batch_size
    nbs = 64    # normal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay；根据accumulate设置超参：权重衰减系数；0.0005
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups    将模型参数分为三组(bn, weights, bias)来进行分组优化
    for k, v in model.named_modules():      # model.named_modules()     https://www.jianshu.com/p/a4c745b6ea9b
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # bias
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)    # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)    # apply decay

    # momentum、nesterov    https://blog.csdn.net/weixin_37958272/article/details/107146938
    # https://www.bilibili.com/video/BV1FT4y1E74V?p=66&vd_source=9c2b9b14820d6f6ec6ccc022af406252
    # 选择优化器，并设置pg0(bn参数)的优化方式
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)  # 默认使用
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})     # add pg1 with weight_decay；设置pg1(weights)的优化方式
    optimizer.add_param_group({'params': pg2})  # add pg2 (bias)    # 设置pg2(biases)的优化方式
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler(学习率调度器) https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    # https://blog.csdn.net/weixin_44751294/article/details/125170729
    # https://blog.csdn.net/qq_43391414/article/details/122869505
    # 自定义学习率函数lf，学习率cos形状变化。(lr: 0.01 ----- epochs -----> 0.002)
    lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_schedule.LambdaLR(optimizer, lr_lambda=lf)

    # EMA，shadow模型，模型指数加权移动平均
    # https://zhuanlan.zhihu.com/p/68748778
    ema = ModelEMA(model)

    start_epoch, best_fitness = 0, 0.0

    # Epochs
    start_epoch = ckpt['epoch'] + 1  # -1 + 1 = 0

    del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)   # grid size (max stride)    获取模型最大stride=32   [32, 16, 8]
    nl = model.model[-1].nl     # nl: number of detection layers  3
    # 获取训练图片和测试图片的分辨率 imgsz=640    imgsz_test=640
    # 执行两次check_img_size()，结果分别赋值给imgsz, imgsz_test
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs,
                                            hyp=hyp, augment=True, cache=False, rect=False, workers=opt.workers,
                                            prefix=colorstr('train: '))
    nb = len(dataloader)    # number of batches  188

    # Process 0
    # 如果报虚拟内存不足(OSError: [WinError 1455] 页面文件太小,无法完成操作)，可降低opt.workers数量
    testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs,
                                   hyp=hyp, cache=False, rect=True, workers=opt.workers,
                                   pad=0.5, prefix=colorstr('val: '))[0]

    if not opt.resume:
        labels = np.concatenate(dataset.labels, 0)      # dataset.labels: 所有img的gt框信息，每张img的格式为(gt_num, cls+xywh(normalized))
        c = torch.tensor(labels[:, 0])  # classes
        if plots:
            plot_labels(labels, names, save_dir, loggers)
            if tb_writer:
                tb_writer.add_histogram('classes', c, 0)

        # Anchors
        # 计算默认锚框anchors与gt框的高宽比
        # gt框的高h宽w与anchors的高h_a宽w_a的比值，即h/h_a，w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
        # 如果bpr小于98%，则根据k-mean算法聚类出新的anchors锚框
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)   # 当前'hat'、'person'数据集，无需重新锚框

        # 不清楚这行代码的意义。手动指定半精度float16后，立即指定回全精度float32，这样做是没意义的
        # 更推荐使用`torch.cuda.amp`模块的自动混合精度，而不是手动指定半精度或全精度。前者可以在不牺牲数值稳定性的前提下，加速你的训练过程。
        # model.half().float()

    # 一些训练要用的参数
    hyp['box'] *= 3. / nl   # box iou损失系数   0.05
    hyp['cls'] *= nc / 80. * 3. / nl    # cls分类损失系数   0.0125
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # obj loss gain   1.0
    hyp['label_smoothing'] = opt.label_smoothing    # 0.0
    model.nc = nc   # attach number of classes to model
    model.hyp = hyp     # attach hyperparameters to model
    model.gr = 1.0      # iou loss ratio (obj_loss = 1.0 or iou)  用于loss计算
    # 从训练样本标签得到类别权重（和类别中的目标数即类别频率成反比）
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc   # tensor(1.86377, 0.13623)
    model.names = names  # 类别名

    # Start training
    t0 = time.time()
    # warmup策略      https://www.zhihu.com/question/338066667
    # 一开始使用较小的学习率，避免一开始学偏了的权重，后面想拉都拉不回来
    # 对于大batch size和大learning rate，学习率呈“上升--平稳--下降”的规律
    # 获取warmup的次数iterations.  max(3 epochs, 1k iterations)
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)

    maps = np.zeros(nc)  # mAP per class

    # https://zhuanlan.zhihu.com/p/56961620
    # https://blog.csdn.net/lucas161543228/article/details/116091424
    # https://blog.csdn.net/ruyingcai666666/article/details/109670567
    # mAP: 平均精确度(average precision)的平均(mean)，是object detection中模型性能的衡量标准。
    # 真正例: TP = TruePositive
    # 真反例: TN = TrueNegative
    # 假正例: FP = FalsePositive
    # 假反例: FN = FalseNegative
    # 查准率: Precision = TP / (TP + FP)   指在所有预测为正例中真正例的比率，也即预测的准确性。
    # 查全率: Recall = TP / (TP + FN)      指在所有正例中被正确预测的比率，也即预测正确的覆盖率。
    # 交并比(IoU)
    # GT(Ground Truth): 相当于待检测物体的label，也就是作为标准答案的框
    # DT(Detect Truth): 我们的模型预测的框
    # mAP@0.5: mean Average Precision(IoU=0.5)，即将IoU设为0.5时，计算每一类的所有图片的AP，然后所有类别求平均，即mAP
    # AP50，AP60，AP70......指取IoU阈值大于0.5，大于0.6，大于0.7......的结果
    # mAP@.5:.95 (mAP@[.5:.95])     表示在不同IoU阈值(从0.5到0.95，步长0.05)(0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95)上的平均mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1      # do not move

    # PyTorch自动混合精度     https://zhuanlan.zhihu.com/p/165152789
    # 在训练最开始之前，实例化一个GradScaler对象
    # 设置amp混合精度训练   GradScaler + autocast
    scaler = amp.GradScaler(enabled=cuda)

    # 初始化损失函数
    compute_loss = ComputeLoss(model)

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    # 开始训练
    for epoch in range(start_epoch, epochs):
        model.train()

        # 暂未使用图片加权策略    (如果分类效果较差，可考虑使用 2023.9.24)
        if opt.image_weights:
            # Generate indices
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(4, device=device)   # mean losses

        # 进度条，方便展示信息
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%25s' * 8) % ('train_epoch/total_epochs', 'train_gpu_mem_reserved', 'train_box_loss', 'train_obj_loss', 'train_cls_loss', 'train_total_loss', 'train_labels_num', 'train_img_size'))
        # 创建进度条
        pbar = tqdm(pbar, total=nb)  # progress bar

        # 梯度清零
        optimizer.zero_grad()
        # batch
        for i, (imgs, targets, paths, _) in pbar:
            # targets  shape: (n, 6)，即 (num_target(整个batch中所有gt框的个数), img_index(该gt框归属于该batch中的那一张img) + class_index(0:hat, 1:person) + xywh(normalized))

            # ni: 计算当前迭代次数 iteration
            ni = i + nb * epoch     # number integrated batches (since train start）
            # non_blocking=True 数据放入GPU     https://blog.csdn.net/qq_37297763/article/details/116670668
            imgs = imgs.to(device, non_blocking=True).float() / 255.0   # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            # 热身训练(前nw次迭代)，热身训练迭代的次数iteration范围[1:nw]，选取较小的accumulate，学习率以及momentum，慢慢的训练
            if ni <= nw:
                xi = [0, nw]    # x interp
                # np.interp    一维线性插值   https://www.jianshu.com/p/a87691a11ae3
                # 关于accumulate的用法，可见上文对nbs的说明
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    # bias的学习率从0.1下降到基准学习率lr*lf(epoch)，其他参数的学习率增加到lr*lf(epoch)
                    # lf为上面设置的余弦退火衰减函数
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale   # 暂未使用
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward   PyTorch自动混合精度   前向过程(model + loss)  开启autocast
            with amp.autocast(enabled=cuda):
                # pred: (32, 3, 80, 80, 7) (32, 3, 40, 40, 7) (32, 3, 20, 20, 7)
                # (batch size, anchor_num, feature map尺度, feature map尺度, xywh+conf+2个class的预测概率)
                pred = model(imgs)      # forward
                # loss为总损失值，loss_items为一个元组，包含分类损失，置信度损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))   # loss scaled by batch_size

            # Backward  反向传播  将梯度放大防止梯度的underflow(amp混合精度训练)
            scaler.scale(loss).backward()

            # Optimize
            # 模型反向传播accumulate次(iterations)后，再根据累计的梯度更新一次参数
            if ni % accumulate == 0:
                # scaler.step()作用: 先将反向传播时被放大的梯度值unscale回来。
                # 如果梯度的值不是infs或者NaNs，那么就去调用optimizer.step()来更新权重
                # 否则，忽略step调用，从而保证权重不更新(不被破坏)
                scaler.step(optimizer)  # optimizer.step
                # 准备着，看是否需要放大Scaler
                scaler.update()
                # 梯度清零
                optimizer.zero_grad()
                if ema:
                    # 更新ema  (用于更新EMA模型的权重)
                    ema.update(model)

            # 打印Print一些信息，包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等信息
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%25s' * 2 + '%25.4g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)  # 进度条显示以上信息

            # Plot  将前三次迭代的batch标签框在图片中画出来并保存。train_batch0/1/2.jpg
            if plots and ni < 3:
                f = save_dir / f'train_batch{ni}.jpg'  # filename
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
            # wandb 显示信息
            elif plots and ni == 10 and wandb_logger.wandb:
                wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                              save_dir.glob('train*.jpg') if x.exists()]})
            # end batch ---------------------------------------------------------------------------

        # Scheduler，一个epoch训练结束后都要调整学习率(学习率衰减)
        # group中三个学习率(pg0、pg1、pg2)每个都要调整
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        # mAP   将model中的属性赋值给ema    (用于更新EMA模型的非权重属性)
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        # 判断当前epoch是否为最后一轮
        final_epoch = epoch + 1 == epochs

        # Calculate mAP
        if opt.test_model or final_epoch:
            wandb_logger.current_epoch = epoch + 1
            # 测试使用的是ema(指数移动平均，对模型的参数做平均)的模型
            # results: 0: mp (1) iou=0.5时，所有类别最大平均f1时，所有类别的平均precision
            #          1: mr (1) iou=0.5时，所有类别最大平均f1时，所有类别的平均recall
            #          2: map50 (1) 所有类别的平均AP@0.5，即map@0.5
            #          3: map (1) 所有类别的平均AP@0.5:0.95，即map@0.5:0.95
            #          4: val_box_loss (1) 验证集回归损失
            #          5: val_obj_loss (1) 验证集置信度损失
            #          6: val_cls_loss (1) 验证集分类损失
            # maps: (2) 各个类别的AP@0.5:0.95
            # t: {tuple: 3}     0: 前向推理耗费的时间   1: nms耗费的时间   2: 总时间
            results, maps, times = test.test(data_dict,     # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
                                             batch_size=batch_size * 2,  # bs
                                             imgsz=imgsz_test,  # test img size
                                             model=ema.ema,     # ema model
                                             single_cls=False,  # 是否是单类数据集
                                             dataloader=testloader,  # test dataloader
                                             save_dir=save_dir,     # 保存地址 runs/train/exp_n
                                             verbose=nc < 50 and final_epoch,   # 是否打印出每个类别的mAP
                                             plots=plots and final_epoch,  # 是否可视化
                                             wandb_logger=wandb_logger,  # 网页可视化，类似于tensorboard
                                             compute_loss=compute_loss,  # 损失函数(train)
                                             is_coco=is_coco)

        # 将训练结果和验证结果追加写入到result.txt中
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(s + '%25.4g' * 7 % results + '\n')

        # tensorboard与wandb的log记录
        #   train
        #       box_loss: 训练集回归损失
        #       obj_loss: 训练集置信度损失
        #       cls_loss: 训练集分类损失
        #   val_metrics
        #       precision: 验证集iou=0.5时，所有类别最大平均f1时，所有类别的平均precision
        #       recall: 验证集iou=0.5时，所有类别最大平均f1时，所有类别的平均recall
        #       mAP_0.5: 验证集所有类别的平均AP@0.5，即map@0.5
        #       mAP_0.5:0.95: 验证集所有类别的平均AP@0.5:0.95，即map@0.5:0.95
        #   val
        #       box_loss: 验证集回归损失
        #       obj_loss: 验证集置信度损失
        #       cls_loss: 验证集分类损失
        #   param_x
        #       lr0/lr1/lr2: 不同网络参数的lr变化
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',       # train loss
                'val_metrics/precision', 'val_metrics/recall', 'val_metrics/mAP_0.5', 'val_metrics/mAP_0.5:0.95',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',             # val loss
                'param_x/lr0', 'param_x/lr1', 'param_x/lr2']                # params
        for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            if wandb_logger.wandb:
                wandb_logger.log({tag: x})  # W&B

        # Update best mAP。这里的best mAP是[P, R, mAP@.5, mAP@.5-.95]的一个加权值
        # fi = 0.1*mAP@.5 + 0.9*mAP@.5-.95
        fi = fitness(np.array(results).reshape(1, -1))
        if fi > best_fitness:
            best_fitness = fi
        wandb_logger.end_epoch(best_result=best_fitness == fi)

        # Save model
        # 保存带checkpoint的模型用于inference或resuming training
        # 保存模型，还保存了epoch，results，optimizer等信息
        # optimizer将不会在最后一轮完成后保存
        # model保存的是EMA模型
        if opt.save_model or (final_epoch and not opt.evolve):
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

            # 为了能获取中间结果，在源码基础上，我们新增如下代码
            if epoch > 0 and epoch % opt.save_model_granularity == 0:
                model_path = wdir / ('epoch_{0}.pt'.format(str(epoch)))
                torch.save(ckpt, model_path)

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            if wandb_logger.wandb:
                if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                    wandb_logger.log_model(
                        last.parent, opt, epoch, fi, best_model=best_fitness == fi)
            del ckpt

        # end epoch --------------------------------------------------------------------------------------
    # end training --------------------------------------------------------------------------------------------------

    # 打印一些信息
    # 可视化训练结果: results.png、confusion_matrix.png，以及('F1', 'PR', 'P', 'R')曲线变化  日志信息
    if plots:
        plot_results(save_dir=save_dir)  # save as results.png
        if wandb_logger.wandb:
            files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
            wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                          if (save_dir / f).exists()]})

    logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    # Strip optimizers
    # 模型训练完后，strip_optimizer函数将optimizer从ckpt中删除
    # 并对模型进行model.half()，将Float32->Float16 这样可以减少模型大小，提高inference速度
    final = best if best.exists() else last  # final model

    ckpts = [ckpt for ckpt in wdir.rglob('*') if ckpt.is_file()]
    for f in ckpts:
        if f.exists():
            strip_optimizer(f)  # strip optimizers

    # Log the stripped model
    if wandb_logger.wandb and not opt.evolve:
        wandb_logger.wandb.log_artifact(str(final), type='model',
                                        name='run_' + wandb_logger.wandb_run.id + '_model',
                                        aliases=['last', 'best', 'stripped'])
    wandb_logger.finish_run()  # 关闭wandb_logger

    # 释放显存
    torch.cuda.empty_cache()

    return results


# LOCAL_RANK    这个Worker，是这台机器上的第几个Worker   -1
# RANK          这个Worker，是全局第几个Worker         -1
# WORLD_SIZE    总共有几个Worker                      1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 源码中，各参数使用的是"-"，而不是"_"，parser.parse_args()会自动将"-"转换为"_"
    # 举例:
    #   parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    #   opt = parser.parse_args()
    #   加载参数时使用的是`opt.img_size`，而不是`opt.img-size`，后者会报错
    # 但这样很容易让人产生误解，因此，我们在这里统一将"-"替换为"_"
    """
    ----------------------------------------------- 常用参数 ---------------------------------------------------------
    --weights: 初始化的权重文件的路径地址，预训练权重
    --cfg: 模型yaml配置文件的路径地址，包括nc、depth_multiple、width_multiple、anchors、backbone、head等，yolov5模型实现细节
    --data: 数据yaml文件的路径地址，训练集、验证集文件路径
    --hyp: 初始超参数文件路径地址
    --epochs: 训练轮次
    --batch_size: 喂入批次文件的多少
    --img_size: 输入图片尺寸
    --resume: 断点续训，从上次打断的训练结果处接着训练，默认False
    --evolve: 是否进行超参数进化，默认False
    --device: 训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    --workers: dataloader中的最大work数(线程个数)
    """
    parser.add_argument('--weights', type=str, default='./weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s_hat.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./data/hat.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='./data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=150)      # 150
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--img_size', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--resume', default=False, help='resume most recent training')
    parser.add_argument('--evolve', default=False, help='evolve hyperparameters')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 如果内存不足，可尝试更改这里        https://blog.csdn.net/didiaopao/article/details/119954291
    parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')

    """
    ----------------------------------------------- 数据增强参数 -------------------------------------------------------
    --adam: 是否使用adam优化器，默认False(使用SGD)
    --linear_lr: 是否使用linear lr(线性学习率)，默认False，使用cosine lr
    --label_smoothing: 标签平滑增强，默认0.0不增强，要增强一般就设为0.1
    """
    parser.add_argument('--adam', default=False, help='use torch.optim.Adam() optimizer')
    parser.add_argument('--linear_lr', default=False, help='linear LR')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')

    """
    ---------------------------------------------- 其他参数 -----------------------------------------------------------
    --project: 训练结果保存的根目录
    --quad: dataloader取数据时，是否使用collate_fn4代替collate_fn，默认False
    --save_period: Log model after every "save_period" epoch    默认-1 不需要log model信息
    --local_rank：rank为进程编号，-1且gpu=1时不进行分布式，-1且多块gpu使用DataParallel模式  DDP参数，请勿修改。
    --cache_images: 是否提前缓存图片到内存，以加快训练速度，默认False，RAM占用太大了
    --image_weights: 是否使用图片加权策略(selection img to training by class weights) 默认False，不使用
    --save_model: 是否对模型进行保存
    --save_model_granularity: 若save_model=True，则训练多少个epoch对模型进行一次保存，默认50
    --test_model: 是否对模型进行测试，计算mAP
    --rect: 训练集是否采用矩形训练，默认False
    --noautoanchor: 不自动调整anchor，默认False(自动调整anchor)
    --multi_scale: 是否进行多尺度训练，默认False
    """
    parser.add_argument('--project', default='./runs/train', help='save to project/name')
    parser.add_argument('--quad', default=False, help='quad dataloader')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--cache_images', default=False, help='cache images for faster training')
    parser.add_argument('--image_weights', default=False, help='use weighted image selection for training')
    parser.add_argument('--save_model', default=True)
    parser.add_argument('--save_model_granularity', type=int, default=50)
    parser.add_argument('--test_model', default=True)
    parser.add_argument('--rect', default=False, help='rectangular training')
    parser.add_argument('--noautoanchor', default=False, help='disable autoanchor check')
    parser.add_argument('--multi_scale', default=False, help='vary img-size +/- 50%%')

    """
    ------------------------------------------ 三个W&B(wandb)参数 -----------------------------------------------------
    --name：训练结果保存的目录，默认是exp，即runs/train/exp
    --upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    --bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B，默认-1，后面会检测到-1后自动赋值
    --entity: wandb entity，默认None
    """
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--upload_dataset', default=False, help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')

    opt = parser.parse_args()

    opt.world_size = 1  # 单机单卡训练
    opt.global_rank = -1    # 单机单卡训练
    set_logging(opt.global_rank)

    # check_requirements()
    # wandb_run = None

    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)  # 训练结果存储位置，自适应命名

    opt.total_batch_size = opt.batch_size
    device = torch.device('cuda:0')

    # Hyperparameters
    with open(opt.hyp, encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    logger.info(opt)
    prefix = colorstr('tensorboard: ')
    tensorboard_log_path = './' + (Path(opt.save_dir) / 'tensorboard_log').as_posix()
    logger.info(f"{prefix}Start with 'tensorboard --logdir={tensorboard_log_path} --port=...'")
    tb_writer = SummaryWriter(tensorboard_log_path)  # Tensorboard

    train(hyp, opt, device, tb_writer)
