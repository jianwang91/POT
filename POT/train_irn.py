import argparse
import torch
import os
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from data import data_voc
from tool import pyutils, torchutils, indexing
import importlib
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def run(args):
    ir_label_out_dir = os.path.join(args.session_name, 'ir_label')

    path_index = indexing.PathIndex(radius=10, default_size=(args.irn_crop_size // 4, args.irn_crop_size // 4))

    model = getattr(importlib.import_module(args.irn_network), 'AffinityDisplacementLoss')(path_index)

    train_dataset = data_voc.VOC12AffinityDataset(args.train_list,
                                                          label_dir=ir_label_out_dir,
                                                          voc12_root=args.voc12_root,
                                                          indices_from=path_index.src_indices,
                                                          indices_to=path_index.dst_indices,
                                                          hor_flip=True,
                                                          crop_size=args.irn_crop_size,
                                                          crop_method="random",
                                                          rescale=(0.5, 1.5)
                                                          )
    train_data_loader = DataLoader(train_dataset, batch_size=args.irn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.irn_batch_size) * args.irn_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizerSGD([
        {'params': param_groups[0], 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay}
    ], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.irn_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.irn_num_epoches))

        for iter, pack in enumerate(train_data_loader):

            img = pack['img'].cuda(non_blocking=True)
            bg_pos_label = pack['aff_bg_pos_label'].cuda(non_blocking=True)
            fg_pos_label = pack['aff_fg_pos_label'].cuda(non_blocking=True)
            neg_label = pack['aff_neg_label'].cuda(non_blocking=True)

            pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = model(img, True)

            bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
            fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
            pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
            neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

            dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
            dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

            avg_meter.add({'loss1': pos_aff_loss.item(), 'loss2': neg_aff_loss.item(),
                           'loss3': dp_fg_loss.item(), 'loss4': dp_bg_loss.item()})

            total_loss = (pos_aff_loss + neg_aff_loss) / 2 + (dp_fg_loss + dp_bg_loss) / 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 10 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % (
                      avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3'), avg_meter.pop('loss4')),
                      'imps:%.1f' % ((iter + 1) * args.irn_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_est_finish()), flush=True)
        else:
            timer.reset_stage()

    infer_dataset = data_voc.VOC12ImageDataset(args.infer_list,
                                                       voc12_root=args.voc12_root,
                                                       crop_size=args.irn_crop_size,
                                                       crop_method="top_left")
    infer_data_loader = DataLoader(infer_dataset, batch_size=args.irn_batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model.eval()
    print('Analyzing displacements mean ... ', end='')

    dp_mean_list = []

    with torch.no_grad():
        for iter, pack in enumerate(infer_data_loader):
            img = pack['img'].cuda(non_blocking=True)

            aff, dp = model(img, False)

            dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())

        model.module.mean_shift.running_mean = torch.mean(torch.stack(dp_mean_list), dim=0)
    print('done.')

    torch.save(model.module.state_dict(), os.path.join(args.session_name, 'ckpt', 'irn.pth'))
    torch.cuda.empty_cache()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--train_list", default="data/trainaug_voc.txt", type=str)
    parser.add_argument("--infer_list", default="data/train_voc.txt", type=str)
    parser.add_argument("--session_name", default="exp_clip", type=str)
    parser.add_argument("--voc12_root", default="/gpfs/work/int/jiawang21/code/data/VOCdevkit/VOC2012/", type=str)

    parser.add_argument("--irn_network", default="network.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)
    args = parser.parse_args()

    run(args)
