import cv2
import os
import random
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tools_seg.Miou import Pa, calculate_miou
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class TextSupervisedLoss(nn.Module):
    def __init__(self, task_type='segmentation', text_weight=0.3):
        super().__init__()
        self.text_weight = text_weight
        self.task_type = task_type

    def cosine_similarity_loss(self, visual_feats, text_feats):
        v = F.adaptive_avg_pool2d(visual_feats, 1).flatten(1)
        t = text_feats.mean(dim=1)
        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)
        cos_sim = (v * t).sum(dim=-1)
        loss = 1 - cos_sim.mean()
        return loss

    def forward(self, visual_feats, text_feats_list):
        text_loss = 0
        for v, t in zip(visual_feats, text_feats_list):
            text_loss += self.cosine_similarity_loss(v, t)
        text_loss /= len(text_feats_list)
        total_loss = self.text_weight * text_loss
        return total_loss

text_supervised_loss = TextSupervisedLoss(task_type='segmentation', text_weight=0.55)

def write_options(model_savedir, args, best_acc_val):
    aaa = []
    aaa.append(['lr', str(args.lr)])
    aaa.append(['batch', args.batch_size])
    aaa.append(['save_name', args.save_name])
    aaa.append(['seed', args.batch_size])
    aaa.append(['best_val_acc', str(best_acc_val)])
    aaa.append(['warm_epoch', args.warm_epoch])
    aaa.append(['end_epoch', args.end_epoch])
    f = open(model_savedir + 'option' + '.txt', "a")
    for option_things in aaa:
        f.write(str(option_things) + '\n')
    f.close()

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def fit(epoch, epochs, model, trainloader, valloader, device, criterion, optimizer, CosineLR):
    scaler = GradScaler()
    if torch.cuda.is_available():
        model.to('cuda')
    running_loss = 0
    model.train()
    train_pa_whole = 0
    train_iou_whole = 0
    epoch_iou = 0
    seg_loss_total = 0
    align_loss_total = 0
    feat_loss_total = 0
    contrast_loss_total = 0
    for batch_idx, (sampled_batch, names) in enumerate(trainloader):
        images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        imgs, masks_cuda, text = images.cuda(), masks.cuda(), text.cuda()
        imgs = imgs.float()
        with autocast():
            masks_pred, visual_features, text_data = model(imgs, text)
            masks_cuda = masks_cuda.squeeze(1)
            total_loss = text_supervised_loss(visual_features, text_data)
            total_loss = total_loss
            loss1 = criterion(masks_pred, masks_cuda)
            loss = 0.5 * total_loss + loss1
            if loss == 0 or None:
                loss = loss + 1e-4
        masks_cuda_max = torch.max(masks_cuda)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        predicted = masks_pred.argmax(1)
        train_iou = calculate_miou(predicted, masks_cuda, 2)
        train_iou_whole += train_iou.item()
        running_loss += loss.item()
        epoch_iou = train_iou_whole / (batch_idx + 1)
    epoch_loss = running_loss / len(trainloader.dataset)
    val_running_loss = 0
    val_pa_whole = 0
    val_iou_whole = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (sampled_batch, names) in enumerate(valloader):
            images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            imgs, masks_cuda, text = images.cuda(), masks.cuda(), text.cuda()
            imgs = imgs.float()
            masks_pred, visual_features, text_data = model(imgs, text)
            masks_cuda = masks_cuda.squeeze(1)
            total_loss = text_supervised_loss(visual_features, text_data)
            predicted = masks_pred.argmax(1)
            val_iou = calculate_miou(predicted, masks_cuda, 2)
            val_iou_whole += val_iou.item()
            loss1 = criterion(masks_pred, masks_cuda)
            loss = total_loss + loss1
            if loss == 0 or None:
                loss = loss + 1e-4
            val_running_loss += loss.item()
            epoch_val_acc = val_pa_whole / (batch_idx + 1)
            epoch_val_iou = val_iou_whole / (batch_idx + 1)
    epoch_val_loss = val_running_loss / len(valloader.dataset)
    CosineLR.step()
    return epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou