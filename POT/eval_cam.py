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
from clip2 import clip
import numpy as np


import tag_clip as clip2
import cv2
import pickle
from lxml import etree
import math
import torch.nn.functional as F
from utils2 import scoremap2bbox, parse_xml_to_dict, _convert_image_to_rgb, compute_AP, compute_F1, _transform_resize
from clip_text2 import class_names_voc, BACKGROUND_CATEGORY_VOC, class_names_coco, BACKGROUND_CATEGORY_COCO, class_names_coco_stuff182_dict, coco_stuff_182_to_27
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

def mask_attn(logits_coarse, logits, h, w, attn_weight):
    patch_size = 16
    candidate_cls_list = []
    logits_refined = logits.clone()

    logits_max = torch.max(logits, dim=0)[0]

    for tempid, tempv in enumerate(logits_max):
        if tempv > 0:
            candidate_cls_list.append(tempid)
    for ccls in candidate_cls_list:
        temp_logits = logits[:, ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits
        mask = mask.reshape(h // patch_size, w // patch_size)

        box, cnt = scoremap2bbox(mask.detach().cpu().numpy(), threshold=temp_logits.mean(), multi_contour_eval=True)
        aff_mask = torch.zeros((mask.shape[0], mask.shape[1])).to(device)
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, mask.shape[0] * mask.shape[1])
        trans_mat = attn_weight * aff_mask
        logits_refined_ccls = torch.matmul(trans_mat, logits_coarse[:, ccls:ccls + 1])
        logits_refined[:, ccls] = logits_refined_ccls.squeeze()
    return logits_refined


def cwr(logits, logits_max, h, w, image, text_features, model):
    patch_size = 16
    input_size = 224
    stride = input_size // patch_size
    candidate_cls_list = []

    ma = logits.max()
    mi = logits.min()
    step = ma - mi
    if args.dataset == 'cocostuff':
        thres_abs = 0.1
    else:
        thres_abs = 0.5
    thres = mi + thres_abs * step

    for tempid, tempv in enumerate(logits_max):
        if tempv > thres:
            candidate_cls_list.append(tempid)
    for ccls in candidate_cls_list:
        temp_logits = logits[:, ccls]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits > 0.5
        mask = mask.reshape(h // patch_size, w // patch_size)

        horizontal_indicies = np.where(np.any(mask.cpu().numpy(), axis=0))[0]
        vertical_indicies = np.where(np.any(mask.cpu().numpy(), axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0

        y1 = max(y1, 0)
        x1 = max(x1, 0)
        y2 = min(y2, mask.shape[-2] - 1)
        x2 = min(x2, mask.shape[-1] - 1)
        if x1 == x2 or y1 == y2:
            return logits_max

        mask = mask[y1:y2, x1:x2]
        mask = mask.float()
        mask = mask[None, None, :, :]
        mask = F.interpolate(mask, size=(stride, stride), mode="nearest")
        mask = mask.squeeze()
        mask = mask.reshape(-1).bool()

        image_cut = image[:, :, int(y1 * patch_size):int(y2 * patch_size), int(x1 * patch_size):int(x2 * patch_size)]
        image_cut = F.interpolate(image_cut, size=(input_size, input_size), mode="bilinear", align_corners=False)
        cls_attn = 1 - torch.ones((stride * stride + 1, stride * stride + 1))
        for j in range(1, cls_attn.shape[1]):
            if not mask[j - 1]:
                cls_attn[0, j] = -1000

        image_features = model.encode_image_tagclip(image_cut, input_size, input_size, attn_mask=cls_attn)[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        cur_logits = logit_scale * image_features @ text_features.t()
        cur_logits = cur_logits[:, 0, :]
        cur_logits = cur_logits.softmax(dim=-1).squeeze()
        cur_logits_norm = cur_logits[ccls]
        logits_max[ccls] = 0.5 * logits_max[ccls] + (1 - 0.5) * cur_logits_norm

    return logits_max

def clip_predict(image_path, model, text_features, class_names):
    NUM_CLASSES = len(class_names)
    pred_label_id = []
    pil_img = Image.open(image_path)
    array_img = np.array(pil_img)
    ori_height, ori_width = array_img.shape[:2]
    if len(array_img.shape) == 2:
        array_img = np.stack([array_img, array_img, array_img], axis=2)
        pil_img = Image.fromarray(np.uint8(array_img))

    patch_size = 16
    preprocess = _transform_resize(int(np.ceil(int(ori_height) / patch_size) * patch_size),
                                   int(np.ceil(int(ori_width) / patch_size) * patch_size))
    image = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract image features
        h, w = image.shape[-2], image.shape[-1]

        image_features, attn_weight_list = model.encode_image_tagclip(image, h, w, attn_mask=1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]

        attn_vote = torch.stack(attn_weight, dim=0).squeeze()

        thres0 = attn_vote.reshape(attn_vote.shape[0], -1)
        thres0 = torch.mean(thres0, dim=-1).reshape(attn_vote.shape[0], 1, 1)
        thres0 = thres0.repeat(1, attn_vote.shape[1], attn_vote.shape[2])

        if args.dataset == 'cocostuff':
            attn_weight = torch.stack(attn_weight, dim=0)[:-1]
        else:
            attn_weight = torch.stack(attn_weight, dim=0)[8:-1]

        attn_cnt = attn_vote > thres0
        attn_cnt = attn_cnt.float()
        attn_cnt = torch.sum(attn_cnt, dim=0)
        attn_cnt = attn_cnt >= 4

        attn_weight = torch.mean(attn_weight, dim=0)[0]
        attn_weight = attn_weight * attn_cnt.float()

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # torch.Size([1, 197, 81])
        logits = logits[:, 1:, :]
        logits = logits.softmax(dim=-1)
        logits_coarse = logits.squeeze()
        logits = torch.matmul(attn_weight, logits)
        logits = logits.squeeze()
        logits = mask_attn(logits_coarse, logits, h, w, attn_weight)

        logits_max = torch.max(logits, dim=0)[0]
        logits_max = logits_max[:NUM_CLASSES]
        logits_max = cwr(logits, logits_max, h, w, image, text_features, model)

        return logits_max.unsqueeze(0)

        # logits_max = logits_max.cpu().numpy()
        # pred_label_id.append(logits_max)

    # return pred_label_id[0]

    # idx = np.argmax(logits_max, axis=0)
    # for i in range(predictions.shape[0]):
    #     id = idx[i]
    #     print('image {}\tlabel\t{}:\t{}'.format(i, classnames[id], predictions[i, id]))
    #     print('image {}:\t{}'.format(i, [v for v in zip(classnames, predictions[i])]))

def run(args, predict_dir, num_cls):
    preds = []
    masks = []
    n_images = 0
    cluster_num = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("../../CLIP-ES/pretrained_models/ViT-B-16.pt", device=device)  # 载入模型
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

                # clip_predict(fg_img, model, text_features,  class_names)

                # with torch.no_grad():
                #     logits_per_image_nomsk = clip_predict(fg_img_nomsk, model, text_features,  class_names)# 第一个值是图像，第二个是第一个的转置
                #     probs_nmsk = logits_per_image_nomsk.softmax(dim=-1).cpu().numpy()  # 图像对应每一个prompt的概率
                #
                #     logits_per_image = clip_predict(fg_img, model, text_features,  class_names) # 第一个值是图像，第二个是第一个的转置
                #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # 图像对应每一个prompt的概率
                #
                #
                #     idx = np.argmax(probs, axis=1) + 1
                #     if idx == key:
                #         for i in range(cluster_num):
                #             filename_k = filename + str(i + 2)
                #             fg_kmeans_img = os.path.join(predict_dir, "mask_combine", filename_k + ".png")
                #             if os.path.exists(fg_kmeans_img) == False:
                #                 continue
                #
                #             logits_per_image_k = clip_predict(fg_kmeans_img, model, text_features,  class_names)  # 第一个值是图像，第二个是第一个的转置
                #             probs_k = logits_per_image_k.softmax(dim=-1).cpu().numpy()  # 图像对应每一个prompt的概率
                #             idx_k = np.argmax(probs_k, axis=1)
                #
                #             cos_compare_path = os.path.join(predict_dir, "cosine_similarities",
                #                                             filename_k + "_cos_sim.npy")
                #             cos_compare = np.load(cos_compare_path)
                #
                #             # if idx_k == key and probs_k[0, idx_k] < probs[0, idx] and cos_compare[
                #             #     0, 0, idx_k] > 0.75:  # 58.62
                #             # if probs_k[0, key] < probs[0, key] and cos_compare[0, 0, key] > 0.8:
                #             if probs_k[0, key-1] < probs[0, key-1] and cos_compare[0, 0, key] > cos_compare[0, 0, 0] and cos_compare[0, 0, key] > 0.7:
                #             # if cos_compare[0, 0, key] > 0.7:
                #             # if 1:
                #                 # if idx_k == key and probs_k[0, idx_k] < probs_nmsk[0, idx] and cos_compare[
                #                 #         0, 0, idx_k] > 0.3:
                #                 # if idx_k == key and probs_k[0, idx_k] < probs[0, idx] and probs_k[0, idx_k]<0.6:
                #                 #     print(cos_compare)
                #                 position = np.where(np.array(keys) == (idx_k+1))[0]
                #                 cos = os.path.join(predict_dir, "mask_cam", filename_k + "_mask_cam.npy")
                #                 cos_sim = np.load(cos)
                #                 cams[position] = cams[position] + cos_sim
                #                 # cams /= np.max(cams) + 1e-5
                #                 print("aaa:", filename_k, probs_k[0, idx_k], probs[0, idx], probs_nmsk[0, idx], "cos:",
                #                       cos_compare[0, 0, key])
                                # print("aaa:", filename_k, probs_k, probs, probs_nmsk, "cos:",
                                #       cos_compare[0, 0, key])

        # k_mask = np.load(os.path.join(predict_dir, "masknp", pack['name'][0] + '.npy'), allow_pickle=True).item()
        #
        #
        # # 读取 kmeans_msk 和 img_msk
        # kmeans_msk = np.load(os.path.join(predict_dir, pack['name'][0] + '_kmeans_msk.npy'))
        # img_msk = np.load(os.path.join(predict_dir, pack['name'][0] + '_img_msk.npy'))

        # print(cams.shape, keys)

        # print(1/0)

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
    
    
# def run(args, predict_dir, num_cls):
#     preds = []
#     masks = []
#     n_images = 0
#     cluster_num = 3
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("../../CLIP-ES/pretrained_models/ViT-B-16.pt", device=device)  # 载入模型
#     text_language = ["person with clothes,people,human", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#                      'bus', 'car', 'cat', 'chair', 'cow',
#                      'diningtable', 'dog', 'horse', 'motorbike',
#                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
#                         'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign']
#     text = clip.tokenize(text_language).to(device)
# 
# 
#     for iter, pack in tqdm(enumerate(dataloader)):
#         n_images += 1
#         cam_dict = np.load(os.path.join(predict_dir,  "npy", pack['name'][0] + '.npy'), allow_pickle=True).item()
#         cams = cam_dict['IS_CAM'] + cam_dict['IS_CAM1']
#         # cams = cam_dict['IS_CAM']
# 
#         keys = cam_dict['keys']
# 
#         for key in keys:
#             if key != 0:
#                 img_nomsk = pack['name'][0]
#                 fg_img_nomsk = os.path.join("/gpfs/work/int/jiawang21/code/data/VOCdevkit/VOC2012/JPEGImages",
#                                             pack['name'][0] + ".jpg")
# 
#                 image_nomsk = preprocess(Image.open(fg_img_nomsk)).unsqueeze(0).to(device)
#                 filename = pack['name'][0] + '_' + str(key) + '_mask_'
#                 fg_img = os.path.join(predict_dir, "visual2", filename + "1.png")
#                 if os.path.exists(fg_img):
#                     image = preprocess(Image.open(fg_img)).unsqueeze(0).to(device)
#                     with torch.no_grad():
#                         logits_per_image_nomsk, logits_per_text = model(image_nomsk, text)  # 第一个值是图像，第二个是第一个的转置
#                         probs_nmsk = logits_per_image_nomsk.softmax(dim=-1).cpu().numpy()  # 图像对应每一个prompt的概率
# 
#                         logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
#                         probs_1 = logits_per_image.softmax(dim=-1).cpu().numpy()  # 图像对应每一个prompt的概率
#                         idx = np.argmax(probs_nmsk, axis=1)
# 
#                         idx_1 = np.argmax(probs_1, axis=1)
#                         if idx_1 == key and probs_1[0, idx_1] > 0.5:
#                             for i in range(cluster_num):
#                                 filename_k = filename + str(i + 2)
#                                 fg_kmeans_img = os.path.join(predict_dir, "visual2", filename_k + ".png")
#                                 image_kmeans = preprocess(Image.open(fg_kmeans_img)).unsqueeze(0).to(device)
#                                 logits_per_image_k, logits_per_text_k = model(image_kmeans, text)
#                                 probs_k = logits_per_image_k.softmax(dim=-1).cpu().numpy()
#                                 idx_k_cluster = np.argmax(probs_k, axis=1)
#                                 if idx_k_cluster == key and probs_k[0, idx_k_cluster] < probs_1[0, idx]:
#                                     print("aaa:", img_nomsk, probs_nmsk[0, idx], probs_1[0, idx_1], probs_k[0, idx_k_cluster], idx_k_cluster, filename_k)
# 
# 
# 
# 
# 
#                         # idx = np.argmax(probs, axis=1)
#                         # if idx == key:
#                         #     for i in range(cluster_num):
#                         #         filename_k = filename + str(i + 2)
#                         #         fg_kmeans_img = os.path.join(predict_dir, "visual2", filename_k + ".png")
#                         #         image_kmeans = preprocess(Image.open(fg_kmeans_img)).unsqueeze(0).to(device)
#                         #         logits_per_image_k, logits_per_text_k = model(image_kmeans, text)  # 第一个值是图像，第二个是第一个的转置
#                         #         probs_k = logits_per_image_k.softmax(dim=-1).cpu().numpy()  # 图像对应每一个prompt的概率
#                         #         idx_k = np.argmax(probs_k, axis=1)
#                         #         if idx_k == key and probs_k[0, idx_k] > probs[0, idx]:
#                         #             print("aaa:", filename_k, probs_k[0, idx_k], probs[0, idx], probs_nmsk[0, idx])
# 
#         cls_labels = np.argmax(cams, axis=0)
#         cls_labels = keys[cls_labels]
#         preds.append(cls_labels.copy())
# 
#         mask = np.array(Image.open(os.path.join(args.gt_path,  pack['name'][0] + '.png')))
#         masks.append(mask.copy())
# 
#     confusion = calc_semantic_segmentation_confusion(preds, masks)[:num_cls, :num_cls]
# 
#     gtj = confusion.sum(axis=1)
#     resj = confusion.sum(axis=0)
#     gtjresj = np.diag(confusion)
#     denominator = gtj + resj - gtjresj
#     iou = gtjresj / denominator
#     print({'iou': iou, 'miou': np.nanmean(iou)})

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
        # dataset = data_voc.VOC12ImageDataset('data/val_' + args.dataset + '.txt', voc12_root=dataset_root,
        #                                      img_normal=None, to_torch=False)

    elif args.dataset == 'coco':
        args.gt_path = "../ms_coco_14&15/SegmentationClass/train2014/"
        dataset_root = "../ms_coco_14&15/images"
        num_cls = 81
        dataset = data_coco.COCOImageDataset('data/train_' + args.dataset + '.txt', coco_root=dataset_root, img_normal=None, to_torch=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    pyutils.Logger(os.path.join(args.session_name, 'eval_' + args.session_name + '.log'))
    run(args, args.session_name , num_cls)

