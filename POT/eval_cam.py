import numpy as np
import os, torch
import argparse
from PIL import Image
from tool import pyutils
from data import data_voc, data_coco
from tqdm import tqdm
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion
import torchvision.transforms as transforms
import numpy as np


import cv2
import pickle
import math
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"



    

def run(args, predict_dir, num_cls):
    preds = []
    masks = []
    n_images = 0
    cluster_num = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        class_names = class_names_voc + BACKGROUND_CATEGORY_VOC
        model, _ = clip2.load("../../CLIP-ES/pretrained_models/ViT-B-16.pt", device=device)
        text_features = clip2.encode_text_with_prompt_ensemble(model, class_names, device)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for iter, pack in tqdm(enumerate(dataloader)):
        n_images += 1
        cam_dict = np.load(os.path.join(predict_dir, "npy", pack['name'][0] + '.npy'), allow_pickle=True).item()
        cams = cam_dict['IS_CAM'] + cam_dict['IS_CAM1']
        # cams = cam_dict['IS_CAM1']
        keys = cam_dict['keys']

        for key in keys:
            if key != 0:
                img_nomsk = pack['name'][0]
                fg_img_nomsk = os.path.join("/gpfs/work/int/jiawang21/code/data/VOCdevkit/VOC2012/JPEGImages",
                                            pack['name'][0] + ".jpg")
                # image_nomsk = preprocess(Image.open(fg_img_nomsk)).unsqueeze(0).to(device)

                filename = pack['name'][0] + '_' + str(key) + '_mask_'
                fg_img = os.path.join(predict_dir, "mask_combine", filename + "1.png")
                if os.path.exists(fg_img) == False:
                    continue

               

        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

        mask = np.array(Image.open(os.path.join(args.gt_path, pack['name'][0] + '.png')))
        masks.append(mask.copy())

    confusion = calc_semantic_segmentation_confusion(preds, masks)[:num_cls, :num_cls]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print({'iou': iou, 'miou': np.nanmean(iou)})
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--gt_path", default='/gpfs/work/int/jiawang21/code/data/VOCdevkit/VOC2012/SegmentationClass', type=str)
    parser.add_argument('--session_name', default="exp", type=str)
    args = parser.parse_args()

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '/gpfs/work/int/jiawang21/code/data/VOCdevkit/VOC2012'
        num_cls = 21
        dataset = data_voc.VOC12ImageDataset('data/train_' + args.dataset + '.txt', voc12_root=dataset_root, img_normal=None, to_torch=False)
        

    elif args.dataset == 'coco':
        args.gt_path = "../ms_coco_14&15/SegmentationClass/train2014/"
        dataset_root = "../ms_coco_14&15/images"
        num_cls = 81
        dataset = data_coco.COCOImageDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root, img_normal=None, to_torch=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    pyutils.Logger(os.path.join(args.session_name, 'eval_' + args.session_name + '.log'))
    run(args, args.session_name , num_cls)

