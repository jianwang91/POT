import torch, os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import importlib
import numpy as np
from tensorboardX import SummaryWriter
from data import data_voc
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from tool import pyutils, torchutils, visualization, imutils
import random
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
warnings.filterwarnings("ignore")

def validate(model, data_loader):
    val_dataset = data_voc.VOC12SegmentationDataset2('data/train_voc.txt',
                                                     label_dir='/path/to/VOC/VOCdevkit/VOC2012/SegmentationClassAug',
                                                     voc12_root='/path/to/VOC/VOCdevkit/VOC2012',
                                                     crop_size=None,
                                                     crop_method="none",
                                                     )
    val_data_loader = DataLoader(val_dataset, batch_size=1,
                                 shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    print('validating ... ', flush=True, end='')

    model.eval()
    dataset = VOCSemanticSegmentationDataset(split='train',
                                             data_dir='/path/to/VOC/VOCdevkit/VOC2012')
    labels = []
    preds = []
    preds_cls = []

    for i, pack in enumerate(val_data_loader):
        img = pack['img'].cuda()
        label = pack['label_cls'].cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
        label = F.pad(label, (0, 0, 0, 0, 1, 0), 'constant', 1.0)
        N_L, _, H_L, W_L = img.shape
        C_L = label[:, 1:].shape[1]
        n, c_c, h_c, w_c = N_L, C_L + 1, H_L, W_L
        c_c = c_c - 1
        norm_clip = torch.zeros(n, c_c, h_c // 16, w_c // 16).cuda()
        for i_clip in range(n):
            cam_dict = np.load(
                os.path.join("/path/to/CLIPES CAM//CLIP_ES_refined_CAM/cams_71",
                             pack['name'][i_clip] + '.npy'), allow_pickle=True).item()
            cams = cam_dict["attn_highres"]
            keys = cam_dict["keys"]
            cams_clip = cams
            refined_cam_all_scales = torch.tensor(cams_clip).cuda()

            refined_cam_all_scales = F.interpolate(refined_cam_all_scales.unsqueeze(0), norm_clip.shape[2:],
                                                   mode='bilinear')
            refined_cam_all_scales = refined_cam_all_scales.squeeze(0)
            for rj in range(refined_cam_all_scales.shape[0]):
                norm_clip[i_clip, keys[rj]] = refined_cam_all_scales[rj]

        outputs = model.forward(img, label, True, None, norm_clip)
        IS_cam = outputs['cam_add']
        IS_cam = F.interpolate(IS_cam, img.shape[2:], mode='bilinear')
        IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
        cls_labels_bkg = torch.argmax(IS_cam, 1)

        cls_cam = outputs['cam_class']
        cls_cam = F.interpolate(cls_cam, img.shape[2:], mode='bilinear')
        cls_cam = cls_cam / (F.adaptive_max_pool2d(cls_cam, (1, 1)) + 1e-5)
        cls_labels_bkg1 = torch.argmax(cls_cam, 1)

        preds.append(cls_labels_bkg[0].cpu().numpy().copy())
        preds_cls.append(cls_labels_bkg1[0].cpu().numpy().copy())
        gt_label = dataset.get_example_by_keys(i, (1,))[0]
        labels.append(gt_label.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    confusion_cls = calc_semantic_segmentation_confusion(preds_cls, labels)
    gtj_cls = confusion_cls.sum(axis=1)
    resj_cls = confusion_cls.sum(axis=0)
    gtjresj_cls = np.diag(confusion_cls)
    denominator_cls = gtj_cls + resj_cls - gtjresj_cls
    iou_cls = gtjresj_cls / denominator_cls

    print({'iou': iou, 'miou': np.nanmean(iou)})
    print({'iou_cls': iou_cls, 'miou_cls': np.nanmean(iou_cls)})

    model.train()
    return np.nanmean(iou)


def setup_seed(seed):
    print("random seed is set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=6, type=int)
    parser.add_argument("--network", default="network.resnet50_POT", type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--wt_dec", default=1e-4, type=float)
    parser.add_argument("--session_name", default="exp", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--tf_freq", default=20, type=int)
    parser.add_argument("--val_freq", default=300, type=int)
    parser.add_argument("--dataset", default="voc", type=str)
    parser.add_argument("--dataset_root", default="/path/to/VOC//VOCdevkit/VOC2012", type=str)
    parser.add_argument("--seed", default=15, type=int)
    parser.add_argument('--img_root', type=str,
                        default='/path/to/VOC//VOCdevkit/VOC2012/JPEGImages')
    parser.add_argument('--split_file', type=str, default='./voc12/train.txt')
    parser.add_argument('--cam_out_dir', type=str, default='./output/voc12/cams')
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.session_name, exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(args.session_name, 'ckpt'), exist_ok=True)
    pyutils.Logger(os.path.join(args.session_name, args.session_name + '.log'))
    tblogger = SummaryWriter(os.path.join(args.session_name, 'runs'))

    assert args.dataset in ['voc', 'coco'], 'Dataset must be voc or coco in this project.'

    if args.dataset == 'voc':
        dataset_root = '/path/to/VOC/VOCdevkit/VOC2012'
        model = getattr(importlib.import_module(args.network), 'Net')(num_cls=21)
        train_dataset = data_voc.VOC12ClsDataset('data/trainaug_' + args.dataset + '.txt', voc12_root=dataset_root,
                                                 resize=(512, 512),
                                                 )

        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
        max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

        val_dataset = data_voc.VOC12ClsDataset('data/train_' + args.dataset + '.txt', voc12_root=dataset_root)
        val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizerSGD([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    bestiou = 0
    for ep in range(args.max_epoches):

        print('Epoch %d/%d' % (ep + 1, args.max_epoches))

        for step, pack in enumerate(train_data_loader):
            img = pack['img'].cuda()
            n, c, h, w = img.shape
            label = pack['label'].cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)

            valid_mask = pack['valid_mask'].cuda()
            valid_mask[:,1:] = valid_mask[:,1:] * label
            valid_mask_lowres = F.interpolate(valid_mask, size=(h//16, w//16), mode='nearest')

            outputs = model.forward(img, valid_mask_lowres)
            score = outputs['score']
            norm_cam = outputs['cam']
            Proto_cam = outputs['Proto_cam']

            lossCLS = F.multilabel_soft_margin_loss(score, label)

            Proto_cam = Proto_cam / (F.adaptive_max_pool2d(Proto_cam, (1, 1)) + 1e-5)
            lossCon = torch.mean(torch.abs(norm_cam - Proto_cam)) * 10

            losses = lossCLS + lossCon
            avg_meter.add({'lossCLS': lossCLS.item(), 'lossCon': lossCon.item()})

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if (optimizer.global_step - 1) % args.print_freq == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'lossCLS:%.4f' % (avg_meter.pop('lossCLS')),
                      'lossCon:%.4f' % (avg_meter.pop('lossCon')),
                      'imps:%.1f' % ((step + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_est_finish()), flush=True)

                # tf record
                tblogger.add_scalar('lossCLS', lossCLS, optimizer.global_step)
                tblogger.add_scalar('lossCon', lossCon, optimizer.global_step)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], optimizer.global_step)

            if (optimizer.global_step - 1) % args.tf_freq == 0:
                # visualization
                img_8 = visualization.convert_to_tf(img[0])
                norm_cam = F.interpolate(norm_cam,img_8.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                Proto_cam = F.interpolate(Proto_cam,img_8.shape[1:],mode='bilinear')[0].detach().cpu().numpy()
                CAM = visualization.generate_vis(norm_cam, None, img_8, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                Proto_cam = visualization.generate_vis(Proto_cam, None, img_8, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)

                # tf record
                tblogger.add_images('CAM', CAM, optimizer.global_step)
                tblogger.add_images('Proto_cam', Proto_cam, optimizer.global_step)

            if (optimizer.global_step-1) % args.val_freq == 0 and optimizer.global_step > 10:
                miou = validate(model, val_data_loader)
                torch.save({'net':model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'iter_' + str(optimizer.global_step) + '.pth'))
                if miou > bestiou:
                    bestiou = miou
                    torch.save({'net':model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'best.pth'))

        else:
            timer.reset_stage()

    torch.save({'net':model.module.state_dict()}, os.path.join(args.session_name, 'ckpt', 'final.pth'))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
