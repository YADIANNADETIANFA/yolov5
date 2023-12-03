import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path, check_imshow
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized


def detect():
    # source: 测试数据文件(图片或视频)的路径，0 for webcam
    # weights: 模型的权重地址，best.pt
    # view_img: 是否展示预测之后的图片或视频，False
    # save_txt: 是否将预测的框坐标以txt文件格式保存，True
    # img_size: 网络输入图片的大小，640
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # 保存预测后的图片
    save_img = True
    # 是否使用webcam，网页数据
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # 检查当前Path(project) / name是否存在 如果存在就新建新的save_dir 默认exist_ok=False 需要重建
    # 将原先传入的名字扩展成新的save_dir 如runs/detect/exp存在 就扩展成 runs/detect/exp1
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize 初始化日志信息
    set_logging()

    device = torch.device('cuda:0')
    half = True  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    # 载入一些模型参数
    # stride: 模型最大的下采样率 [8, 16, 32]，所有stride一般为32
    stride = int(model.stride.max())  # model stride

    # 确保输入图片的尺寸imgsz能整除stride=32，如果不能则调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16，加速推理

    # Get names and colors
    # 得到当前model模型的所有类的类名
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 加载推理数据
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    # 推理前测试，这里先设置一个全零的Tensor进行一次前向推理，判断程序是否正常
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    time_start = time_synchronized()
    # 正式推理
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # 处理每一张图片/视频的格式
        # path: 图片/视频的路径
        # img: 进行resize + pad之后的图片
        # img0s: 原尺寸的图片
        # vid_cap: 当读取图片时为None，读取视频时为视频源
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 如果图片是3维的(RGB)，就在前面添加一个维度1，batch_size=1
        # 因为输入网络的图片需要是4维的 [batch_size, channel, w, h]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # 对每张图片/视频进行前向推理
        t1 = time_synchronized()
        # pred shape: (1, num_boxes, xywh + obj_conf + classes) = (1, 18900, 7)
        pred = model(img, augment=opt.augment)[0]

        # nms除去多余的框
        # Apply NMS  进行NMS
        # conf_thres: 置信度阈值
        # iou_thres: iou阈值
        # classes: 是否只保留特定的类别 默认为None
        # agnostic_nms: 进行nms是否也去除不同类别之间的框
        # pred: (num_obj, 6) = (5, 6)   这里的预测信息pred还是相对于 img_size(640) 的
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        # 后续保存或者打印预测信息
        # 对每张图片进行处理，将pred(相对img_size 640)映射回原图img0 size
        for i, det in enumerate(pred):  # detections per image
            # 从LoadImages流读取文件中的照片或者视频，batch_size=1
            # p: 当前图片/视频的绝对路径
            # s: 输出信息 初始为 ''
            # im0: 原始图片 letterbox + pad 之前的图片
            # frame: 初始为0  可能是当前图片属于视频中的第几帧？
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # 当前图片路径
            p = Path(p)  # to Path
            # 图片/视频的保存路径save_path
            save_path = str(save_dir / p.name)  # img.jpg

            # txt文件(保存预测框坐标)保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # 我们的数据源是webcam，暂时不需要这里
            # path = Path(txt_path)
            # path.mkdir(parents=True)

            # print string 输出信息，图片shape (w,h)
            s += '%gx%g ' % img.shape[2:]

            # normalization gain gn = [w, h, w, h]  用于后面的归一化
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息(相对img_size 640)映射回原图 img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 输出信息s + 检测到的各个类别的目标个数
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测信息: txt、img0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id+score+xywh
                    if save_txt:  # Write to file(txt)
                        # 将xyxy(左上角 + 右下角)格式转换为xywh(中心的 + 宽高)格式 并除以gn(whwh)做归一化 转为list再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            # 是否需要显示我们预测后的结果  img0(此时已将pred结果可视化到了img0中)
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 是否需要保存图片或视频（检测后的图片/视频 里面已经被我们画好了框的） img0
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                            if opt.save_webcam_video:
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    if opt.save_webcam_video:
                        vid_writer.write(im0)

        time_end = time_synchronized()
        if opt.save_webcam_video and time_end - time_start > opt.save_video_duration:
            break

    # 推理结束，保存结果，打印信息
    # 保存预测的label信息 xywh等   save_txt
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")
    # 打印预测的总时间
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    """
    opt参数解析
    --weights: 模型的权重地址
    --img_size: 网络输入图片的大小 默认640
    --conf_thres: object置信度阈值 默认0.25
    --iou_thres: 做nms的iou阈值 默认0.45
    --device: 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
    --view_img: 是否展示预测之后的图片或视频 默认False
    --save_txt: 是否将预测的框坐标以txt文件格式保存，会在runs/detect/expn/labels下生成每张图片预测的txt文件。默认False
    --save_conf: 是否保存预测每个目标的置信度到预测txt文件中 默认False
    --classes: 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
    --agnostic_nms: 进行nms是否也除去不同类别之间的框 默认False
    --augment: 预测是否也要采用数据增强 TTA
    --project: 当前测试结果放在哪个主文件夹下 默认runs/detect
    --name: 当前测试结果放在run/detect下的文件名  默认是exp
    --source: 测试数据(图片或视频或webcam)的来源。(在这里，我们一般使用webcam进行测试验证)
    --save_webcam_video: 对于source=webcam的情况，是否将视频结果保存下来，默认是True
    --save_video_duration: 对于source=webcam的情况，若将视频结果保存下来，视频的时长设定。默认是30s
    """
    # 源码中，各参数使用的是"-"，而不是"_"，parser.parse_args()会自动将"-"转换为"_"
    # 举例:
    #   parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    #   opt = parser.parse_args()
    #   加载参数时使用的是`opt.img_size`，而不是`opt.img-size`，后者会报错
    # 但这样很容易让人产生误解，因此，我们在这里统一将"-"替换为"_"
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/epoch_50.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view_img', default=False, help='display results')
    parser.add_argument('--save_txt', default=False, help='save results to *.txt')
    parser.add_argument('--save_conf', default=False, help='save confidences in --save-txt labels')
    parser.add_argument('--classes', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', default=False, help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_webcam_video', default=True, help='if source is webcam, whether to save the video result')
    parser.add_argument('--save_video_duration', type=int, default='30', help='time of the saved video, unit is seconds')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()
