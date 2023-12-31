#### yolov5

识别人员是否佩戴安全帽



环境依赖，实测可用

> ./conda_requirement.yaml
>
> ./pip_requirement.txt



其他文件

> `./weights/yolov5s.pt`：预训练权重
>
> `./VOCdevkit/`：训练数据
>
> `./runs/train/exp/`：网络训练时的输出
>
> `./runs/detect/exp2/0.mp4`：网络预测效果
>
> 
>
> (将`./weights/`与`./VOCdevkit/`下载后放到项目文件下即可。)
>
> (`./runs/train/exp/`与`./runs/detect/exp2/0.mp4`为示例效果，仅用于参考，不是项目运行所需文件)
>
> 
>
> 链接：https://pan.baidu.com/s/1AeuJjHp3QAg-6Eck0hvxww?pwd=1ked 
> 提取码：1ked 



网络训练

```bash
python train.py
```



网络效果验证

```bash
python detect.py
```



网络训练过程中，输出文件介绍

> **labels_correlogram.jpg**
>
> + 所有gt框xywh多变量联合分布直方图
>
> + 出自`utils/plots.py中的plot_labels()`
>
> + 于wandb可见，于本地项目目录可见
>
> **labels.jpg**
>
> + 所有gt框的类别分布、所有gt框形状、所有gt框的xy直方图、所有gt框的wh直方图
>
> + 出自`utils/plots.py中的plot_labels()`
>
> + 于wandb可见，于本地项目目录可见
>
> **train_batch0/1/2.png**
>
> + 训练阶段，前三个batch的图片分别拼接成一个4*4的mosaic大图并保存，并标注出gt框
>
> + 出自`train.py中的plot_images()`
>
> + 于wandb可见，于本地项目目录可见
>
> **results.txt**
>
> + 记录训练结果和验证结果
> + 出自`train.py`
> + 于本地项目目录可见
>
> **test_batch0/1/2_labels.jpg**
>
> + final_epoch，val验证时，前三个batch的图片及其gt_labels，拼成一个4*4大图并保存
> + 出自`test.py`中的`Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()`
> + 于本地项目目录可见
>
> **test_batch0/1/2_pred.jpg**
>
> + final_epoch，val验证时，前三个batch的图片及其pred的结果，拼成一个4*4大图并保存
> + 出自`test.py`中的`Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()`
> + 于wandb可见，于本地项目目录可见
>
> **PR_curve.png**
>
> + final_epoch，val验证时，pr曲线
> + 出自`test.py`中的`p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)`中的`plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)`
> + 于wandb可见，于本地项目目录可见
>
> **F1_curve.png**
>
> + 同上，F1_conf曲线
>
> **P_curve.png**
>
> + 同上，P_conf曲线
>
> **R_curve.png**
>
> + 同上，R_conf曲线
>
> **confusion_matrix.png**
>
> + final_epoch，val验证时，混淆矩阵
> + 出自`test.py`中的`confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))`
> + 于wandb可见，于本地项目目录可见
>
> **result.png**
>
> + 训练结束后，将result.txt中的某些列的数据(10/14)，展示出来
> + 出自`train.py`中的`plot_results()`
> + 于wandb可见，于本地项目目录可见
>
> 
>
> 其他内容：
>
> **所有gt框类别分布：train.py中的`tb_writer.add_histogram('classes', c, 0)`**
>
> + 于tensorboard可见
>
> **Bounding Box Debugger/Images**
>
> + 在`test.py`中进行测试时，当前的测试效果
> + 出自`test.py`
> + 于wandb可见
>
> **train系列、val_metrics系列、val系列、param_x系列**
>
> + 在`train.py`中，记录train和val时的各项指标变化
> + 出自`train.py`
> + 于tensorboard和wandb可见



网络效果展示

<img src="https://icarustypora.oss-cn-shenzhen.aliyuncs.com/AI/yolo/yolov5_%E7%BD%91%E7%BB%9C%E6%95%88%E6%9E%9C%E5%B1%95%E7%A4%BA_person.png" alt="yolov5_网络效果展示_person" style="zoom: 25%;" />



<img src="https://icarustypora.oss-cn-shenzhen.aliyuncs.com/AI/yolo/yolov5_%E7%BD%91%E7%BB%9C%E6%95%88%E6%9E%9C%E5%B1%95%E7%A4%BA_hat.png" alt="yolov5_网络效果展示_hat(没有安全帽，用钢盆替代一下)" style="zoom:25%;" />



关于数据标注

> labelimg工具;
>
> 我们标出来的是.xml格式，左上角为(0, 0)，右下角为(width_max, height_max);
>
> .xml中，width为横向长度，height为纵向长度。
>
> 
>
> .txt为经由.xml转换而来；
>
> 每一行代表一个标注框的数据;
>
> 每行5个数。第一个数为类别(0:hat, 1:person)；第二、三个数为中心点坐标的归一化值(width、height)；第四、五个数为宽高的归一化值(width、height)。
>
> 
>
> `./VOCdevkit/VOC2007/`：全部数据集，数量7581
>
> `./VOCdevkit/images/train/`：训练集图片，数量5991
>
> `./VOCdevkit/images/val`：验证集图片，数量1590
>
> 5991 + 1590 = 7581
>
> 
>
> [详参，CSDN大佬"炮哥带你学"](https://blog.csdn.net/didiaopao/article/details/119954291?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170150265316800185828363%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170150265316800185828363&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-119954291-null-null.142^v96^pc_search_result_base1&utm_term=%E7%82%AE%E5%93%A5%E5%B8%A6%E4%BD%A0%E5%AD%A6yolo5&spm=1018.2226.3001.4187)



github仓库

> https://github.com/YADIANNADETIANFA/yolov5

个人博客

> http://www.icarus.wiki/archives/60bde475.html

wandb训练记录

> https://wandb.ai/icarus_athena/YOLOv5/overview?workspace=user-icarus_athena



参考链接

> [B站大佬"同济子豪兄"](https://www.bilibili.com/video/BV15w411Z7LG?p=1&vd_source=9c2b9b14820d6f6ec6ccc022af406252)
>
> [CSDN大佬"满船清梦压星河HK"](https://blog.csdn.net/qq_38253797/article/details/119043919)
>
> [CSDN大佬"炮哥带你学"](https://blog.csdn.net/didiaopao/article/details/119954291?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170150265316800185828363%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170150265316800185828363&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-119954291-null-null.142^v96^pc_search_result_base1&utm_term=%E7%82%AE%E5%93%A5%E5%B8%A6%E4%BD%A0%E5%AD%A6yolo5&spm=1018.2226.3001.4187)

<font color="FF0000">感谢以上大佬的分享(如有遗漏请告知，我这边进行补充)</font>
