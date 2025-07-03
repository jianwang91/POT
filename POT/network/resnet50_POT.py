import torch, os
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50
from torch.cuda.amp import autocast
import cv2
from PIL import Image
import numpy as np
import torch
from kmeans_pytorch import kmeans
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings

warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):

    def __init__(self, num_cls=21):
        super(Net, self).__init__()

        self.num_cls = num_cls

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))
        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.side1 = nn.Conv2d(256 + 3, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512 + 3, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024 + 3, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048 + 3, 256, 1, bias=False)
        self.classifier = nn.Conv2d(2048, self.num_cls - 1, 1, bias=False)
        self.f9 = torch.nn.Conv2d(2 + 3 + 2048, 2048, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.f9, self.side1, self.side2, self.side3, self.side4])


    def Sinkhorn(self, K, u, v):
            r = torch.ones_like(u)
            c = torch.ones_like(v)
            thresh = 1e-2
            max_iter = 100
            for i in range(max_iter):
                r0 = r
                r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
                c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
                err = (r - r0).abs().mean()
                if err.item() < thresh:
                    break

            T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

            return T

    def get_seed_aff_x4(self, norm_cam, label, f, feature):
            n, c, h, w = norm_cam.shape
            seeds = torch.zeros((n, h, w, c)).cuda()
            belonging = norm_cam.argmax(1)
            seeds = seeds.scatter_(-1, belonging.view(n, h, w, 1), 1).permute(0, 3, 1, 2).contiguous()
            seeds = seeds * label

            n, c_f, h, w = feature.shape
            seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
            num_classes = c
            num_clusters = 2
            all_cam_maps = torch.zeros(n, num_classes, num_clusters, h, w).cuda()
            cluster_centers_tensor = torch.zeros(n, num_classes, num_clusters, c_f).cuda()

            for batch_idx in range(n):
                for i in range(num_classes):
                    mask = seeds[batch_idx, i, :, :].unsqueeze(0)
                    class_features = feature[batch_idx] * mask  #  c_f * h * w
                    class_features = class_features.view(c_f, -1).transpose(0, 1)  # (h * w, c_f)
                    valid_mask = mask.view(-1) > 0  # (h * w,)
                    valid_features = class_features[valid_mask]  # (num_valid_pixels, c_f)

                    if valid_features.size(0) < num_clusters:
                        continue

                    cluster_ids, cluster_centers = kmeans(
                        X=valid_features, num_clusters=num_clusters, distance='cosine',
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    )

                    cluster_centers_tensor[batch_idx, i, :, :] = cluster_centers
                    cluster_centers = cluster_centers.unsqueeze(1)
                    full_features = feature[batch_idx].view(c_f, -1).transpose(0, 1)  #(h * w, c_f)
                    full_features = full_features.unsqueeze(0)  #  (1, h*w, c_f)

                    cosine_sims = F.cosine_similarity(full_features, cluster_centers.cuda(),
                                                      dim=2)  # (num_clusters, h*w)
                    cosine_sims = cosine_sims.view(num_clusters, h, w)  # (num_clusters, h, w)

                    cam_map = cosine_sims.mean(dim=0)  #  (h, w)
                    all_cam_maps[batch_idx, i] = cam_map

                feature_ot = feature[batch_idx].clone().view(c_f, -1).permute(1, 0).contiguous()  # HW * n * C
                feature_ot = feature_ot / (torch.norm(feature_ot, dim=1, keepdim=True) + 1e-5)
                cluster_centers_tensor_ot = cluster_centers_tensor.clone().view(n, c * num_clusters, c_f)[
                    batch_idx].contiguous()
                cluster_centers_tensor_ot = cluster_centers_tensor_ot / (
                            torch.norm(cluster_centers_tensor_ot, dim=1, keepdim=True) + 1e-5)

                sim = torch.einsum('md,nd->mn', feature_ot, cluster_centers_tensor_ot).contiguous()
                M = h * w
                N = num_clusters
                n_cls = c
                eps = 0.1
                sim = sim.view(M, N, n_cls)
                sim = sim.permute(2, 0, 1)
                wdist = 1.0 - sim
                xx = torch.zeros(n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
                yy = torch.zeros(n_cls, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)

                with torch.no_grad():
                    KK = torch.exp(-wdist / eps)
                    T = self.Sinkhorn(KK, xx, yy)
                if torch.isnan(T).any():
                    return None

                T = T.cuda()
                T = T.permute(0, 2, 1).view(1, num_classes, num_clusters, h, w)

                thres = 0.2
                mean = 0.5
                cam_mask = (all_cam_maps[batch_idx] > (mean - thres)) & (all_cam_maps[batch_idx] < (mean + thres))

            all_cam_maps = all_cam_maps.mean(dim=2) * label
            all_cam_maps = all_cam_maps / (F.adaptive_max_pool2d(all_cam_maps, (1, 1)) + 1e-5)

            return all_cam_maps



    @autocast()
    def forward(self, x, valid_mask, test=True, seed=None, norm_clip=None, filenames=None):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(torch.cat([x1, F.interpolate(x, x1.shape[2:], mode='bilinear', align_corners=True)], 1))
        side2 = self.side2(torch.cat([x2, F.interpolate(x, x2.shape[2:], mode='bilinear', align_corners=True)], 1))
        side3 = self.side3(torch.cat([x3, F.interpolate(x, x3.shape[2:], mode='bilinear', align_corners=True)], 1))
        side4 = self.side4(torch.cat([x4, F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True)], 1))

        hie_fea = torch.cat(
            [F.interpolate(side1 / (torch.norm(side1, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
             F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
             F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
             F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear')],
            dim=1)

        cam = self.classifier(x4)
        score = F.adaptive_avg_pool2d(cam, 1)

        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True) * valid_mask

        cam_clip_bkg = 1 - torch.max(norm_clip, dim=1)[0].unsqueeze(1)
        norm_clip = torch.cat([cam_clip_bkg, norm_clip], dim=1)
        norm_cam_ori = norm_cam.clone()
        norm_cam = norm_cam * valid_mask
        similarity_f = torch.cat([F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True), x4], dim=1)


        cam_class = self.get_seed_aff_x4(norm_cam, valid_mask, similarity_f, hie_fea)
        norm_clip = F.interpolate(norm_clip, norm_cam.shape[2:], mode='bilinear', align_corners=True)


        c_norm_clip = norm_clip.shape[1] - 1
        clip_bg_expand = norm_clip[:, 0:1, :, :].expand(-1, c_norm_clip, -1, -1)
        cam_class = cam_class[:, 1:, :, :]
        cam_class = F.relu(cam_class - 0.5 * clip_bg_expand)
        cam_class_bkg = 1 - torch.max(cam_class, dim=1)[0].unsqueeze(1)
        cam_class = torch.cat([cam_class_bkg, cam_class], dim=1)

        norm_clip = norm_clip[:, 1:, :, :]
        norm_clip = F.relu(norm_clip - 0.15 * clip_bg_expand)
        norm_clip_bkg = 1 - torch.max(norm_clip, dim=1)[0].unsqueeze(1)
        norm_clip = torch.cat([norm_clip_bkg, norm_clip], dim=1)
        norm_clip = norm_clip * valid_mask

        norm_clip = F.interpolate(norm_clip, norm_cam.shape[2:], mode='bilinear', align_corners=True)

        sum_channels = norm_clip.sum(dim=1, keepdim=True)
        norm_clip = F.relu(1.2 * norm_clip - 0.2 * sum_channels)
        norm_clip = norm_clip / (F.adaptive_max_pool2d(norm_clip, (1, 1)) + 1e-5)

        norm_clip_mean = norm_clip.mean(dim=3, keepdim=True)
        mask = norm_clip < 0.5 * norm_clip_mean
        norm_clip[mask] = 0

        norm_clip_classes = torch.argmax(norm_clip, dim=1, keepdim=True)  # (N, 1, H, W)
        cam_class_classes = torch.argmax(cam_class, dim=1, keepdim=True)  # (N, 1, H, W)
        class_match = norm_clip_classes == cam_class_classes  # (N, C, H, W)
        matched_add = torch.where(class_match, norm_clip + 0.2 * cam_class, norm_clip)

        result = matched_add
        result[result > 1] = 1
        return {"score": score, "cam": norm_cam,
                "feature": hie_fea, "norm_cam_ori": norm_cam_ori, "cam_class": cam_class, "cam_add": result
                }

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, num_cls):
        super(CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    @autocast()
    def forward(self, x, label, cams_clip, keys, packs=None):
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(torch.cat([x1, F.interpolate(x, x1.shape[2:], mode='bilinear', align_corners=True)], 1))
        side2 = self.side2(torch.cat([x2, F.interpolate(x, x2.shape[2:], mode='bilinear', align_corners=True)], 1))
        side3 = self.side3(torch.cat([x3, F.interpolate(x, x3.shape[2:], mode='bilinear', align_corners=True)], 1))
        side4 = self.side4(torch.cat([x4, F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True)], 1))

        cam = self.classifier(x4)
        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        x = (x[0] + x[1].flip(-1)).unsqueeze(0)
        x2 = (x2[0] + x2[1].flip(-1)).unsqueeze(0)
        x4 = (x4[0] + x4[1].flip(-1)).unsqueeze(0)
        x3 = (x3[0] + x3[1].flip(-1)).unsqueeze(0)

        hie_fea = torch.cat(
            [F.interpolate(side1 / (torch.norm(side1, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
             F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
             F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
             F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear')],
            dim=1)

        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)

        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)
        similarity_f = torch.cat([F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True), x4], dim=1)

        n, c_c, _, _ = norm_cam.shape
        h_c, w_c = x.shape[-2] // 16, x.shape[-1] // 16
        c_c = c_c - 1

        norm_clip = torch.zeros(n, c_c, h_c, w_c).cuda()
        keys = torch.tensor(keys)
        cams_clip = torch.tensor(cams_clip).cuda()
        cams_clip = F.interpolate(cams_clip.unsqueeze(0), norm_clip.shape[2:], mode='bilinear', align_corners=True)
        refined_cam_all_scales = cams_clip.squeeze(0)
        for rj in range(refined_cam_all_scales.shape[0]):
            norm_clip[0, keys[rj]] = refined_cam_all_scales[rj]

        cam_clip_bkg = 1 - torch.max(norm_clip, dim=1)[0].unsqueeze(1)
        norm_clip = torch.cat([cam_clip_bkg, norm_clip], dim=1)

        norm_clip = F.interpolate(norm_clip, norm_cam.shape[2:], mode='bilinear', align_corners=True)
        cam_class = self.get_seed_aff_x4(norm_cam, label.unsqueeze(0), similarity_f, hie_fea)
        c_norm_clip = norm_clip.shape[1] - 1
        clip_bg_expand = norm_clip[:, 0:1, :, :].expand(-1, c_norm_clip, -1, -1)

        cam_class = cam_class[:, 1:, :, :]
        cam_class = F.relu(cam_class - 0.5 * clip_bg_expand)

        cam_class_bkg = 1 - torch.max(cam_class, dim=1)[0].unsqueeze(1)
        cam_class = torch.cat([cam_class_bkg, cam_class], dim=1)


        sum_channels = norm_clip.sum(dim=1, keepdim=True)
        norm_clip = F.relu(1.2 * norm_clip - 0.2 * sum_channels)
        norm_clip = norm_clip / (F.adaptive_max_pool2d(norm_clip, (1, 1)) + 1e-5)
        norm_clip_classes = torch.argmax(norm_clip, dim=1, keepdim=True)  # (N, 1, H, W)
        cam_class_classes = torch.argmax(cam_class, dim=1, keepdim=True)  # (N, 1, H, W)
        class_match = norm_clip_classes == cam_class_classes  # (N, C, H, W)
        matched_add = torch.where(class_match, norm_clip + 0.3 * cam_class, norm_clip)
        result = matched_add
        result[result > 1] = 1
        result = result.clone()
        n, _, h, w = result.shape


        return cam_class[0], result[0]







