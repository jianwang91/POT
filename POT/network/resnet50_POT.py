import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50
from kmeans_pytorch import kmeans


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

        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)
        self.classifier = nn.Conv2d(2048, self.num_cls-1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.side1, self.side2, self.side3, self.side4])

    def get_seed(self, norm_cam, label, feature):
        n,c,h,w = norm_cam.shape

        # iou evalution
        seeds = torch.zeros((n,h,w,c)).cuda()
        feature_s = feature.view(n,-1,h*w)
        feature_s = feature_s/(torch.norm(feature_s,dim=1,keepdim=True)+1e-5)
        correlation = F.relu(torch.matmul(feature_s.transpose(2,1), feature_s),inplace=True).unsqueeze(1) #[n,1,h*w,h*w]
        cam_flatten = norm_cam.view(n,-1,h*w).unsqueeze(2) #[n,21,1,h*w]
        inter = (correlation * cam_flatten).sum(-1)
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter/union).view(n,self.num_cls,h,w) #[n,21,h,w]
        miou[:,0] = miou[:,0]*0.5
        probs = F.softmax(miou, dim=1)
        belonging = miou.argmax(1)
        seeds = seeds.scatter_(-1, belonging.view(n,h,w,1), 1).permute(0,3,1,2).contiguous()
        
        seeds = seeds * label
        return seeds, probs
    
    def get_prototype(self, seeds, feature):
        n,c,h,w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1)  # seed:[n,21,1,h,w], feature:[n,1,c,h,w], crop_feature:[n,21,c,h,w]
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1,c,h,w), (1,1)).view(n, self.num_cls, c, 1, 1) # prototypes:[n,21,c,1,1]
        return prototype

    def reactivate(self, prototype, feature):
        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2)) # feature:[n,1,c,h,w], prototypes:[n,21,c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam

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

    def forward(self, x, valid_mask):

        N, C, H, W = x.size()

        # forward
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        sem_feature = x4
        cam = self.classifier(x4)
        score = F.adaptive_avg_pool2d(cam, 1)

        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)*valid_mask

        
        seeds, probs = self.get_seed(norm_cam.clone(), valid_mask.clone(), sem_feature.clone())
        prototypes = self.get_prototype(seeds, hie_fea)
        IS_cam = self.reactivate(prototypes, hie_fea)
        IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)

        similarity_f = torch.cat([F.interpolate(x, x4.shape[2:], mode='bilinear', align_corners=True), x4], dim=1)
        cam_class = self.get_seed_aff_x4(IS_cam, valid_mask, similarity_f,
                                                              hie_fea)

        cam_class = cam_class[:, 1:]
        cam_class_bkg = 1 - torch.max(cam_class, dim=1)[0].unsqueeze(1)
        cam_class = torch.cat([cam_class_bkg, cam_class], dim=1)

        return {"score": score, "cam": norm_cam, "Proto_cam": cam_class}

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

    def forward(self, x, label):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())        
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        cam = self.classifier(x4)
        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)
        
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

        seeds, _ = self.get_seed(norm_cam.clone(), label.unsqueeze(0).clone(), hie_fea.clone())
        prototypes = self.get_prototype(seeds, hie_fea)
        Proto_cam = self.reactivate(prototypes, hie_fea)


        return norm_cam[0], Proto_cam[0]
