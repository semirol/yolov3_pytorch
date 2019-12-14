from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size,img_path,anno_path):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size,\
         augment=False, multiscale=False,img_path=img_path,anno_path=anno_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/final_weight.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--txt_path", type=str, default='', help="test_txt_path,default for valid")
    parser.add_argument("--img_path", type=str, default='data/custom/images', help="img_dir_path,default for valid")
    parser.add_argument("--anno_path", type=str, default='data/custom/labels', help="anno_dir_path,default for valid")
    opt = parser.parse_args()
    print(opt)

    if opt.txt_path != '':
        txt_list=[]
        # transform test.txt
        with open(opt.txt_path,"r") as file:
            f1 = open('tmp/tmp_test.txt',"w")
            img_list = file.readlines()
            txt_list = img_list
            for line in img_list:
                line = os.path.join(opt.img_path,line.strip()+'.jpg')
                f1.write(line+'\n')
            f1.close()

        # transform *.txt which are annos
        print('Transform annos to local format...')
        count=0
        for fname in tqdm.tqdm(txt_list,desc='Transforming'):
            f = open(os.path.join(opt.anno_path,fname.strip()+'.txt'),encoding='utf-8')
            fw = open(os.path.join('tmp/tmp_anno',fname.strip()+'.txt'),'w',encoding='utf-8')
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                line = line[1:]
                if line[0]!='带电芯充电宝' and line[0]!='不带电芯充电宝':
                    continue
                
                if line[0]=='带电芯充电宝':
                    fw.write('0 ')
                elif line[0]=='不带电芯充电宝':
                    fw.write('1 ')
                    
                xmin = int(line[1])
                ymin = int(line[2])
                xmax = int(line[3])
                ymax = int(line[4])
                
                tmppath = os.path.join(opt.img_path,fname).strip('\n')+'.jpg'

                image = Image.open(tmppath)
                width = int(image.size[0])
                height = int(image.size[1])
                
                fw.write(str((xmin+xmax)/2/width))
                fw.write(' ')
                fw.write(str((ymin+ymax)/2/height))
                fw.write(' ')
                fw.write(str((xmax-xmin)/width))
                fw.write(' ')
                fw.write(str((ymax-ymin)/height))
                fw.write('\n')

            f.close()
            fw.close()

        ##

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    if opt.txt_path == '':
        valid_path = data_config["valid"]
        anno_path = opt.anno_path
    else:
        valid_path = 'tmp/tmp_test.txt'
        anno_path = 'tmp/tmp_anno/'
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        img_path=opt.img_path,
        anno_path=anno_path,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
