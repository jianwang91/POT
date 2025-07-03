from builtins import bool
import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import importlib
import os
import imageio
import argparse
from data import data_voc, data_coco
from tool import torchutils, pyutils
import warnings

warnings.filterwarnings("ignore")
import clip
from clip_text import class_names, new_class_names, BACKGROUND_CATEGORY  # , imagenet_templates
from pytorch_grad_cam import GradCAM
from lxml import etree
from utils import parse_xml_to_dict, scoremap2bbox
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
import cv2
from pytorch_grad_cam.utils.image import scale_cam_image
import time

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings

warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

device = "cuda" if torch.cuda.is_available() else "cpu"


def overlap(img, hm):
    hm = plt.cm.jet(hm)[:, :, :3]
    hm = np.array(
        Image.fromarray((hm * 255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(
        np.float) * 2
    if hm.shape == np.array(img).astype(np.float).shape:
        out = (hm + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(hm.shape)
        print(np.array(img).shape)
    return out


def draw_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    gt_cat = np.where(gt_label == 1)[0]
    for _, gt in enumerate(gt_cat):
        heatmap = overlap(orig_img, norm_cam[gt])
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))
        imageio.imsave(cam_viz_path, heatmap)


class ClipOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with cuda.device(process_id):
        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            label = F.pad(label, (1, 0), 'constant', 1.0)

            cam_dict = np.load(
                os.path.join("/gpfs/work/int/jiawang21/code/data/CLIP_ES_refined_CAM/cams_71",
                             img_name + '.npy'), allow_pickle=True).item()
                # os.path.join("/gpfs/work/int/jiawang21/code/CLIP-ES/output/voc12/cams",
                #              img_name + '.npy'), allow_pickle=True).item()
            cams = cam_dict["attn_highres"]
            keys = cam_dict["keys"]
            # bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 2)
            # cams_clip = np.concatenate((bg_score, cams), axis=0)
            cams_clip = cams

            outputs = [model(img[0].cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1),
                             cams_clip, keys, pack) for img in pack['img']]

            # multi-scale fusion
            IS_CAM_list = [output[1].cpu() for output in outputs]
            IS_CAM_list = [IS_CAM_list[0]]
            IS_CAM_list = [F.interpolate(torch.unsqueeze(o, 1), size, mode='bilinear', align_corners=False) for o in
                           IS_CAM_list]
            IS_CAM = torch.sum(torch.stack(IS_CAM_list, 0), 0)[:, 0]
            IS_CAM /= F.adaptive_max_pool2d(IS_CAM, (1, 1)) + 1e-5
            IS_CAM = IS_CAM.detach().numpy()



            # save IS_CAM


            IS_CAM_list1 = [output[0].cpu() for output in outputs]  # 4 * torch.Size([21, 18, 32])
            IS_CAM_list1 = [F.interpolate(torch.unsqueeze(o.float(), 1), size, mode='bilinear', align_corners=False) for
                            o in IS_CAM_list1]  # torch.Size([21, 1, 281, 500])
            IS_CAM1 = torch.sum(torch.stack(IS_CAM_list1, 0), 0)[:, 0]  # torch.Size([21, 281, 500])
            # print(valid_cat)
            # if valid_cat
            # IS_CAM1 = F.softmax(IS_CAM1, dim=0)
            IS_CAM1 /= F.adaptive_max_pool2d(IS_CAM1, (1, 1)) + 1e-5  # torch.Size([21, 281, 500])
            IS_CAM1 = IS_CAM1.detach().numpy()
            IS_CAM1 = IS_CAM1 + np.pad(IS_CAM1[1:, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.4)
            # IS_CAM1 = IS_CAM1.detach().numpy()
            # save IS_CAM


            # visualize IS-CAM
            if args.visualize:
                orig_img = np.array(Image.open(pack['img_path'][0]).convert('RGB'))
                draw_heatmap((IS_CAM1 + IS_CAM).copy(), label, orig_img, os.path.join(args.session_name, 'visual'), img_name)

            valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
            IS_CAM = IS_CAM[valid_cat]
            valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
            IS_CAM1 = IS_CAM1[valid_cat]

            np.save(os.path.join(args.session_name, 'npy', img_name + '.npy'),
                    {"keys": valid_cat, "IS_CAM": IS_CAM, "IS_CAM1": IS_CAM1})

            # 释放共享的 CUDA 张量
            torch.cuda.empty_cache()

            # 收集和释放共享的 CUDA 张量
            torch.cuda.ipc_collect()

            # if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
            #     print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_SIPE", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--ckpt", default="final.pth", type=str)
    parser.add_argument("--visualize", default=False, type=bool)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument('--model_clip', type=str,
                        default='/gpfs/work/int/jiawang21/code/CLIP-ES/pretrained_models/ViT-B-16.pt')

    args = parser.parse_args()
    print("start make cam", time.strftime('%Y:%m:%d %H:%M:%S', time.localtime(time.time())))

    os.makedirs(os.path.join(args.session_name, 'npy'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'visual'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, 'infer.log'))
    print(vars(args))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '/gpfs/work/int/jiawang21/code/data/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=21)
        dataset = data_voc.VOC12ClsDatasetMSF('data/train_' + args.dataset + '.txt', voc12_root=dataset_root,
        # dataset = data_voc.VOC12ClsDatasetMSF('data/test_' + args.dataset + '.txt', voc12_root=dataset_root,
                                              # scales=(1.0, 2.0))
                                              scales=(1.0, 0.5, 1.5, 2.0))
    #     trainaug_   val_

    elif args.dataset == 'coco':
        dataset_root = "../ms_coco_14&15/images"
        model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=81)
        dataset = data_coco.COCOClsDatasetMSF('data/train_' + args.dataset + '_tmp.txt', coco_root=dataset_root,
                                              scales=(1.0, 0.5, 1.5, 2.0))

    checkpoint = torch.load(args.session_name + '/ckpt/' + args.ckpt)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()
    # n_gpus = 1

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    print("end make cam", time.strftime('%Y:%m:%d %H:%M:%S', time.localtime(time.time())))

    torch.cuda.empty_cache()


# from builtins import bool
# import torch
# from torch import multiprocessing, cuda
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from torch.backends import cudnn
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import importlib
# import os
# import imageio
# import argparse
# from data import data_voc, data_coco
# from tool import torchutils, pyutils
# import warnings
#
# warnings.filterwarnings("ignore")
# import clip
# from clip_text import class_names, new_class_names, BACKGROUND_CATEGORY  # , imagenet_templates
# from pytorch_grad_cam import GradCAM
# from lxml import etree
# from utils import parse_xml_to_dict, scoremap2bbox
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
# import cv2
# from pytorch_grad_cam.utils.image import scale_cam_image
# import time
#
# try:
#     from torchvision.transforms import InterpolationMode
#
#     BICUBIC = InterpolationMode.BICUBIC
# except ImportError:
#     BICUBIC = Image.BICUBIC
# import warnings
#
# warnings.filterwarnings("ignore")
# _CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
#
# def overlap(img, hm):
#     hm = plt.cm.jet(hm)[:, :, :3]
#     hm = np.array(
#         Image.fromarray((hm * 255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(
#         np.float) * 2
#     if hm.shape == np.array(img).astype(np.float).shape:
#         out = (hm + np.array(img).astype(np.float)) / 3
#         out = (out / np.max(out) * 255).astype(np.uint8)
#     else:
#         print(hm.shape)
#         print(np.array(img).shape)
#     return out
#
#
# def draw_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
#     gt_cat = np.where(gt_label == 1)[0]
#     for _, gt in enumerate(gt_cat):
#         heatmap = overlap(orig_img, norm_cam[gt])
#         cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))
#         imageio.imsave(cam_viz_path, heatmap)
#
#
# class ClipOutputTarget:
#     def __init__(self, category):
#         self.category = category
#
#     def __call__(self, model_output):
#         if len(model_output.shape) == 1:
#             return model_output[self.category]
#         return model_output[:, self.category]
#
#
# def _work(process_id, model, dataset, args):
#     databin = dataset[process_id]
#     n_gpus = torch.cuda.device_count()
#     data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
#
#     with cuda.device(process_id):
#         model.cuda()
#
#         for iter, pack in enumerate(data_loader):
#             img_name = pack['name'][0]
#             label = pack['label'][0]
#             size = pack['size']
#             label = F.pad(label, (1, 0), 'constant', 1.0)
#
#             cam_dict = np.load(
#                 os.path.join("/gpfs/work/int/jiawang21/code/data/CLIP_ES_refined_CAM/cams_71",
#                              img_name + '.npy'), allow_pickle=True).item()
#                 # os.path.join("/gpfs/work/int/jiawang21/code/CLIP-ES/output/voc12/cams",
#                 #              img_name + '.npy'), allow_pickle=True).item()
#             cams = cam_dict["attn_highres"]
#             keys = cam_dict["keys"]
#             # bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 2)
#             # cams_clip = np.concatenate((bg_score, cams), axis=0)
#             cams_clip = cams
#
#             outputs = [model(img[0].cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1),
#                              cams_clip, keys, pack) for img in pack['img']]
#
#             # multi-scale fusion
#             IS_CAM_list = [output[1].cpu() for output in outputs]
#             IS_CAM_list = [IS_CAM_list[0]]
#             IS_CAM_list = [F.interpolate(torch.unsqueeze(o, 1), size, mode='bilinear', align_corners=False) for o in
#                            IS_CAM_list]
#             IS_CAM = torch.sum(torch.stack(IS_CAM_list, 0), 0)[:, 0]
#             IS_CAM /= F.adaptive_max_pool2d(IS_CAM, (1, 1)) + 1e-5
#             IS_CAM = IS_CAM.detach().numpy()
#
#
#
#             # save IS_CAM
#
#
#             IS_CAM_list1 = [output[0].cpu() for output in outputs]  # 4 * torch.Size([21, 18, 32])
#             IS_CAM_list1 = [F.interpolate(torch.unsqueeze(o.float(), 1), size, mode='bilinear', align_corners=False) for
#                             o in IS_CAM_list1]  # torch.Size([21, 1, 281, 500])
#             IS_CAM1 = torch.sum(torch.stack(IS_CAM_list1, 0), 0)[:, 0]  # torch.Size([21, 281, 500])
#             # print(valid_cat)
#             # if valid_cat
#             # IS_CAM1 = F.softmax(IS_CAM1, dim=0)
#             IS_CAM1 /= F.adaptive_max_pool2d(IS_CAM1, (1, 1)) + 1e-5  # torch.Size([21, 281, 500])
#             IS_CAM1 = IS_CAM1.detach().numpy()
#             IS_CAM1 = IS_CAM1 + np.pad(IS_CAM1[1:, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.4)
#             # IS_CAM1 = IS_CAM1.detach().numpy()
#             # save IS_CAM
#
#
#             # visualize IS-CAM
#             if args.visualize:
#                 orig_img = np.array(Image.open(pack['img_path'][0]).convert('RGB'))
#                 draw_heatmap((IS_CAM).copy(), label, orig_img, os.path.join(args.session_name, 'visual'), img_name)
#
#             valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
#             IS_CAM = IS_CAM[valid_cat]
#             valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
#             IS_CAM1 = IS_CAM1[valid_cat]
#
#             np.save(os.path.join(args.session_name, 'npy', img_name + '.npy'),
#                     {"keys": valid_cat, "IS_CAM": IS_CAM, "IS_CAM1": IS_CAM1})
#
#             # 释放共享的 CUDA 张量
#             torch.cuda.empty_cache()
#
#             # 收集和释放共享的 CUDA 张量
#             torch.cuda.ipc_collect()
#
#             # if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
#             #     print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--network", default="network.resnet50_SIPE", type=str)
#     parser.add_argument("--num_workers", default=0, type=int)
#     parser.add_argument("--session_name", default="exp", type=str)
#     parser.add_argument("--ckpt", default="final.pth", type=str)
#     parser.add_argument("--visualize", default=True, type=bool)
#     parser.add_argument("--dataset", default="voc", type=str)
#     parser.add_argument('--model_clip', type=str,
#                         default='/gpfs/work/int/jiawang21/code/CLIP-ES/pretrained_models/ViT-B-16.pt')
#
#     args = parser.parse_args()
#     print("start make cam", time.strftime('%Y:%m:%d %H:%M:%S', time.localtime(time.time())))
#
#     os.makedirs(os.path.join(args.session_name, 'npy'), exist_ok=True)
#     os.makedirs(os.path.join(args.session_name, 'visual'), exist_ok=True)
#     pyutils.Logger(os.path.join(args.session_name, 'infer.log'))
#     print(vars(args))
#
#     assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'
#
#     if args.dataset == 'voc':
#         dataset_root = '/gpfs/work/int/jiawang21/code/data/VOCdevkit/VOC2012'
#         model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=21)
#         dataset = data_voc.VOC12ClsDatasetMSF('data/train_' + args.dataset + '.txt', voc12_root=dataset_root,
#         # dataset = data_voc.VOC12ClsDatasetMSF('data/test_' + args.dataset + '.txt', voc12_root=dataset_root,
#                                               # scales=(1.0, 2.0))
#                                               scales=(1.0, 0.5, 1.5, 2.0))
#     #     trainaug_   val_
#
#     elif args.dataset == 'coco':
#         dataset_root = "../ms_coco_14&15/images"
#         model = getattr(importlib.import_module(args.network), 'CAM')(num_cls=81)
#         dataset = data_coco.COCOClsDatasetMSF('data/train_' + args.dataset + '_tmp.txt', coco_root=dataset_root,
#                                               scales=(1.0, 0.5, 1.5, 2.0))
#
#     checkpoint = torch.load(args.session_name + '/ckpt/' + args.ckpt)
#     model.load_state_dict(checkpoint['net'], strict=True)
#     model.eval()
#
#     n_gpus = torch.cuda.device_count()
#     # n_gpus = 1
#
#     dataset = torchutils.split_dataset(dataset, n_gpus)
#
#     print('[ ', end='')
#     multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
#     print(']')
#
#     print("end make cam", time.strftime('%Y:%m:%d %H:%M:%S', time.localtime(time.time())))
#
#     torch.cuda.empty_cache()