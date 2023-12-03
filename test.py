from pathlib import Path
from threading import Thread
import numpy as np
import torch
from tqdm import tqdm
from utils.general import box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import time_synchronized


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.25,      # for NMS
         iou_thres=0.45,       # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,
         save_hybrid=False,
         save_conf=False,
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False):
    """
    :params data: 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息。由train.py调用时传入data_dict
    :params weights: 模型的权重文件地址。由train.py调用时为None，直接运行test.py时默认weights/yolov5s.pt
    :params batch_size: 前向传播的批次大小。直接运行test.py时传入默认32，由train.py调用时则传入batch_size // WORLD_SIZE * 2
    :params imgsz: 输入网络的图片分辨率。直接运行test.py时传入默认640，由train.py调用时传入imgsz_test
    :params conf_thres: object置信度阈值，默认0.001
    :params iou_thres: 进行NMS时IOU的阈值，默认0.6
    :params task: 设置测试的类型，有train, val, test, speed or study几种，默认val
    :params device: 测试的设备
    :params single_cls: 数据集是否只用一个类别，直接运行test.py时传入默认False，由train.py调用时传入single_cls参数
    :params augment: 测试是否使用TTA(Test Time Augment)，默认False
    :params verbose: 是否打印出每个类别的mAP，直接运行test.py时传入默认Fasle，由train.py调用时传入nc < 50 and final_epoch
    :params save_txt: 是否以txt文件的形式保存模型预测框的坐标
    :params save_hybrid: 是否save label+prediction hybrid results to *.txt，默认False
    :params save_conf: 是否保存预测每个目标的置信度到预测txt文件中
    :params save_json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）
                       直接运行test.py时传入默认False，由train.py调用时则传入is_coco and final_epoch(一般也是False)
    :params project: 测试保存的源文件，默认runs/test
    :params name: 测试保存的文件地址，默认exp，保存在runs/test/exp下
    :params exist_ok: 默认False，一般都要重新创建文件夹
    :params half: 是否使用半精度推理，FP16 half-precision inference
    :params model: 模型，直接执行test.py时为None，由train.py调用时传入ema.ema(ema模型)
    :params dataloader: 数据加载器，直接执行test.py时为None，由train.py调用时传入testloader
    :params save_dir: 文件保存路径，直接执行test.py时为''，由train.py调用时传入save_dir(runs/train/exp_n)
    :params plots: 是否可视化，直接运行test.py时传入默认True，由train.py调用时传入plots and final_epoch
    :params wandb_logger: 网页可视化，类似于tensorboard，直接运行test.py时传入默认None，由train.py调用时传入wandb_logger(train)
    :params compute_loss: 损失函数，直接运行test.py时传入默认None，由train.py调用时传入compute_loss(train)
    :return (Precision, Recall, map@0.5, map@0.5:0.95, box_loss, obj_loss, cls_loss)
    """
    # Initialize/load model and set device
    training = model is not None
    device = torch.device('cuda:0')

    # 使用half，不但模型需要设为half，输入模型的图片也需要设为half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # 启用模型验证模式
    model.eval()

    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # 计算mAP相关参数
    # 设置iou阈值 从0.5-0.95取10个(0.05间隔)   iou vector for mAP@0.5:0.95
    # iouv: [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # mAP@0.5:0.95 iou个数=10个
    niou = iouv.numel()

    # Logging 初始化日志
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)      # 16


    # 初始化一些测试需要的参数
    seen = 0    # 初始化测试的图片的数量
    # 初始化混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取数据集所有类别的类名
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    # 设置tqdm进度条的显示信息
    # Class: 类别
    # Images_num: 数据集图片数量
    # Labels_num: 数据集gt框的数量
    # Precision: iou=0.5时，所有类别最大平均f1时，所有类别的平均precision
    # Recall: iou=0.5时，所有类别最大平均f1时，所有类别的平均recall
    # mAP@.5: 所有类别的平均AP@0.5，即map@0.5
    # mAP@.5:.95': 所有类别的平均AP@0.5:0.95，即map@0.5:0.95
    s = ('%25s' + '%25s' * 6) % ('val_Class', 'val_Images_num', 'val_Labels_num', 'val_Precision', 'val_Recall', 'val_mAP@.5', 'val_mAP@.5:.95')

    # 初始化p, r, f1, mp, mr, map50, map指标和时间t0, t1
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # 初始化测试集的损失
    loss = torch.zeros(3, device=device)
    # 初始化json文件中的字典，统计信息，ap等
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    # 开始验证
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)     # non_blocking=True 数据放入GPU
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # targets  shape: (n, 6)，即 (num_target(整个batch中所有gt框的个数), img_index(该gt框归属于该batch中的哪一张img) + class_index(0:hat, 1:person) + xywh(normalized))
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model 前向推理
            t = time_synchronized()
            # augment: yolo.py   class Model(nn.Module): forward    暂未用到该参数
            # out:       (bs, 3个Detect layer(feature map)的anchor_num * grid_w * grid_h堆叠在一起, xywh+conf+classes) = (64, 11088+2772+693, 7)
            # train_out: 一个tensor list，存放三个元素，(bs, anchor_num, grid_w, grid_h, xywh+conf+classes)
            #                    如: (64, 3, 80, 80, 7) (64, 3, 40, 40, 7) (64, 3, 20, 20, 7)      (img size不一定是640*640，所以不一定是80/40/20的grid_w和grid_h，但是下采样倍数的关系还是一致的，如44/22/11，84/42/21)
            out, train_out = model(img, augment=augment)
            # 累计前向推理时间
            t0 += time_synchronized() - t

            # Compute loss  计算验证集损失
            # compute_loss不为空，说明正在执行train.py，根据传入的compute_loss计算损失值
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # lbox, lobj, lcls

            # Run NMS   (Non-Maximum Suppression, NMS) 非极大值抑制   如字面意思，就是抑制不是极大值的元素，可以理解为局部最大搜索
            # 将真实框target(gt)的xywh(因target是在labeling中做了归一化)映射到img(test)尺寸
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            # save_hybrid: adding the dataset labels to the model predictions before NMS
            #              是在NMS之前将数据集标签targets添加到模型预测中
            # 这允许在数据集中自动标记(for autolabelling)其他对象(在pred中混入gt) 并且mAP反映了新的混合标签
            # targets: (num_target, img_index+class_index+xywh) = (31, 6)
            # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
            # 不过这里没用到
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling   一般save_hybrid为False

            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            # 累计NMS时间
            t1 += time_synchronized() - t

        # Statistics per image  统计每张图片的真实框、预测框信息
        # 为每张图片做统计，写入预测信息到txt文件，生成json文件字典，统计tp等
        # out: 一个list，每个元素是该batch中每张img的结果。     每张img的结果: (符合条件的预测框的个数, xyxy + 确定是物体的条件下，判定为某个class类的概率 + 该class类的index(0/1))，即(n, 6)
        for si, pred in enumerate(out):
            # 获取第si张图片的gt框标签信息，包括class_index(0:hat, 1:person)+xywh(normalized);   target[:,0]: 该gt框归属于该batch中的哪一张img
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)    # 第si张图片的gt个数
            # 获取标签类别
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])  # 第si张图片的地址
            seen += 1   # 统计测试图片数量+1

            if len(pred) == 0:
                if nl:
                    # 如果第si张图片的预测为空，且存在gt框，则添加空的信息到stats里
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            # 将预测坐标映射到原图img中
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # 保存预测信息到txt文件，默认False   runs\test\exp7\labels\image_name.txt
            if save_txt:
                # gn = [w, h, w, h] 对应图片的宽高，用于后面归一化
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    # xyxy -> xywh 并作归一化处理
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # 保存预测类别和坐标值到对应图片image_name.txt文件中
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            # 保存预测信息到wandb_logger(类似tensorboard)中
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))  # 后面会将图片上传到wandb，命名：Bounding Box Debugger/Images
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Assign all predictions as incorrect
            # 计算混淆矩阵，计算correct，生成stats
            # 初始化预测评定 niou为iou阈值的个数
            # correct: (pred预测框个数, 10(iouv.shape))  全部预设为False。 correct存储的是，所有预测框，在各个iou等级上，最大能达到哪里(能到达的iou等级为True，不能到达的iou等级为False)
            #
            # correct(TP)计算过程
            # for gt中所有类别:
            # 1、选出pred中属于该类别的所有预测框
            # 2、选出gt中属于该类别的所有gt框
            # 3、计算出选出的所有预测框 和 选出的所有gt框 ious
            # 4、筛选出所有ious > 0.5的预测框，就是TP
            # 5、如果存在TP就统计所有TP中不同iou阈值下的TP，同时统计检测到的目标(detected)
            # 6、重复这个过程，直到检测到的目标个数len(detected)=gt个数
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:  # 第si张img的gt框个数
                detected = []  # target indices     用于存放已检测到的目标
                tcls_tensor = labels[:, 0]  # 第si张img的所有gt框的class_index(0:hat, 1:person)

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])    # 第si张img的所有gt框，获取xyxy格式的数据
                # 将预测框映射到原图img
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    # 计算混淆矩阵 confusion_matrix
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class  对img中每个类别(0:hat, 1:person)单独处理
                for cls in torch.unique(tcls_tensor):
                    # gt中该类别的索引 target indices  nonzeros: 获取列表中为True的index
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # 预测框中该类别的索引    prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # predn(pi, :4): 属于某个类别的预测框；    tbox[ti]: 属于某个类别的gt框
                        # box_iou: 计算属于该类的预测框与属于该类的gt框的iou
                        # .max(1): ious: 选出每个预测框与所有gt框的iou中，最大的iou值; i: 为最大iou值时对应的gt框索引
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):  # j: ious中大于0.5的索引  只有iou>=0.5才是TP
                            d = ti[i[j]]  # detected target     成功获得检测到的目标
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)      # 将当前检测到的gt框d添加到detected()
                                # iouv为以0.05为步长  0.5-0.95的序列
                                # 统计所有TP中不同iou阈值下的TP  true positive  并在correct中记录下哪个预测框是哪个iou阈值下的TP
                                # correct: (pred_num, 10) = (300, 10)  记录着哪个预测框在哪个iou阈值下是TP。注意，correct中各个元素为布尔值
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    # all targets already located in image     如果已检测到的目标个数，等于gt框的个数，就结束
                                    break

            # 将当前这张图片的预测结果统计到stats中
            # Append statistics (correct, conf, pcls, tcls)
            # correct: (pred_num, 10) 当前图片每一个预测框在每一个iou条件下是否是TP。各元素为bool
            # pred[:, 4]: (pred_num, 1) 当前图片每一个预测框的conf
            # pred[:, 5]: (pred_num, 1) 当前图片每一个预测框的类别
            # tcls: (gt_num, 1) 当前图片所有gt框的class
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images   画出前三个batch的图片的ground truth和预测框predictions(两个图)一起保存
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    # 计算mAP
    # 统计stats中所有图片的统计结果，将stats列表的信息拼接到一起
    # stats(concat后): list{4} correct, conf, pcls, tcls
    # correct (img_sum, 10) 整个数据集所有图片中所有预测框在每一个iou条件下是否是TP  (476459, 10)，各元素为bool
    # conf (img_sum) 整个数据集所有图片中所有预测框的conf  (476459)
    # pcls (img_sum) 整个数据集所有图片中所有预测框的类别   (476459)
    # tcls (gt_sum) 整个数据集所有图片所有gt框的class     (25022)
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():   # any()，判断集合中是否存在可迭代对象
        # 根据上面的统计预测结果，计算p, r, ap, f1, ap_class指标
        # p: (nc=2) iou=0.5时，所有类别最大平均f1时，各个类别的precision
        # r: (nc=2) iou=0.5时，所有类别最大平均f1时，各个类别的recall
        # ap: (nc=2, 10) 各个类别分别在10个iou下的AP
        # f1: (nc=2) iou=0.5时，所有类别最大平均f1时，各类别分别的f1值
        # ap_class: (nc=2) 数据集中所有的类别index
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)

        # ap50: (nc=2) 各个类别的AP@0.5   ap: (nc=2) 各个类别的AP@0.5:0.95
        ap50, ap = ap[:, 0], ap.mean(1)


        # mp: (1) iou=0.5时，所有类别最大平均f1时，所有类别的平均precision
        # mr: (1) iou=0.5时，所有类别最大平均f1时，所有类别的平均recall
        # map50: (1) 所有类别的平均AP@0.5，即map@0.5
        # map: (1) 所有类别的平均AP@0.5:0.95，即map@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

        # nt: (nc) 统计出，整个数据集所有图片的所有gt框的，各个类别的出现次数
        # np.bincount: 计算非负数组中每个值出现的次数      https://blog.csdn.net/xlinsist/article/details/51346523
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class    nt:ndarray(2), (1971, 23051)  可以看到，我们的数据集，不同类别的样本量，并不是很均衡
    else:
        nt = torch.zeros(1)

    # 打印各项指标
    # Print results
    #       seen:       数据集图片数量
    #       nt.sum():   数据集gt框的数量
    #       mp:         (1) iou=0.5时，所有类别最大平均f1时，所有类别的平均precision
    #       mr:         (1) iou=0.5时，所有类别最大平均f1时，所有类别的平均recall
    #       map50:      (1) 所有类别的平均AP@0.5，即map@0.5
    #       map:        (1) 所有类别的平均AP@0.5:0.95，即map@0.5:0.95
    pf = '%25s' + '%25i' * 2 + '%25.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 细节展示每个类别的各个指标
    #       names[c]:   这个类别
    #       seen:       数据集图片数量
    #       nt[c]:      这个类别的gt框数量
    #       p[i]:       iou=0.5时，所有类别最大平均f1时，这个类别的precision
    #       r[i]:       iou=0.5时，所有类别最大平均f1时，这个类别的recall
    #       ap50[i]:    这个类别的AP@0.5
    #       ap[i]:      这个类别的AP@0.5:0.95
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds  打印前向推理耗费的时间、nms耗费的时间、总时间
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots     画出混淆矩阵，并存入wandb_logger中
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()): {tuple:7}
    #      0: (1) iou=0.5时，所有类别最大平均f1时，所有类别的平均precision
    #      1: (1) iou=0.5时，所有类别最大平均f1时，所有类别的平均recall
    #      2: (1) 所有类别的平均AP@0.5，即map@0.5
    #      3: (1) 所有类别的平均AP@0.5:0.95，即map@0.5:0.95
    #      4: val_box_loss (1) 验证集回归损失
    #      5: val_obj_loss (1) 验证集置信度损失
    #      6: val_cls_loss (1) 验证集分类损失
    # maps: (2) 各个类别的AP@0.5:0.95
    # t: {tuple: 3}     0: 前向推理耗费的时间   1: nms耗费的时间   2: 总时间
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
