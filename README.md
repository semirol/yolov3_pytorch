# yolov3_pytorch
根据 https://github.com/eriklindernoren/PyTorch-YOLOv3 的版本进行了修改，并使用新的数据集进行了训练。

## 安装
##### git clone & 安装依赖
    $ git clone https://github.com/semirol/yolov3_pytorch
    $ cd yolov3_pytorch
    $ pip install -r requirements.txt

##### 下载预训练权重
    将下载的final_weight.pth放入weights/文件夹：
        百度云链接：https://pan.baidu.com/s/1kLWiTaK5RtEw1niaD20PLQ
        提取码：z72f

## 测试

    $ python3 test.py --txt_path *.txt --img_path imgdir/ --anno_path annodir/
    
    测试数据格式和训练集完全相同。
    
    "--batch_size", type=int, default=8, help="size of each image batch"
    "--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file"
    "--data_config", type=str, default="config/custom.data", help="path to data config file"
    "--weights_path", type=str, default="weights/final_weight.pth", help="path to weights file"
    "--class_path", type=str, default="data/custom.names", help="path to class label file"
    "--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected"
    "--conf_thres", type=float, default=0.001, help="object confidence threshold"
    "--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression"
    "--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation"
    "--img_size", type=int, default=416, help="size of each image dimension"
    
    "--txt_path", type=str, default='', help="test_txt_path,default for valid"
    "--img_path", type=str, default='data/custom/images', help="img_dir_path,default for valid"
    "--anno_path", type=str, default='data/custom/labels', help="anno_dir_path,default for valid"
    
    后三个参数在default模式下仅用于验证，但由于安全问题训练验证集已删除，故如需测试则必须重新指定这三个路径。
    若测试出现问题请发邮件至350395090@qq.com或加微信dongshanwei001。
