import cv2
import os
import random
import torch
import copy
import time
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils import data
import numpy as np
import pandas as pd
# from tools_mine.produse_label import produce_label
from fit_MLAT import fit,set_seed,write_options
from sklearn import metrics
from dataset.Load_Dataset_AMDSD import RandomGenerator, ValGenerator, ImageToImage2D, LV2D
from dataset.create_dataset_rgb import for_train_transform,test_transform
import argparse
import warnings
import segmentation_models_pytorch as smp
# from models.deeplabv3.deeplabv3_model import DeepLabV3
# from models.unet.unet_model import Unet
# from duibi.unetplusplus import UNetplusplus
import torch.backends.cudnn as cudnn
# from duibi.segunet.segUnet import ModSegNet
# from duibi.deeplabv3_AFMA.deeplabv3_model import My_DeepLabV3
# from duibi.medicalpolartraining.mpt import UNet
# from duibi.resunet.Resunet import ResUnet
from duibi.r2unet.R2unet import R2AttU_Net
# from duibi.hnet.H_Net import CloverNet
import ConfigAMDSD as config
from utils import read_text
from duibi.attu.attunet import AttU_Net
from duibi.msgse_net.msgsenet import MSGSE_4_ds
# from duibi.unetxt.UNetxt import UNext
from duibi.egeunet.EGEUnet import EGEUNet
from duibi.cenet.CEnet import CE_Net
# from duibi.msnet.Msnet import M2SNet
# from duibi.MCNet.MCnet import net_factory
from duibi.transu.vit_seg_modeling import VisionTransformer as ViT_seg
from duibi.transu.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from duibi.brau.braunet import BRAUnet
from duibi.resunet.Resunet import ResUnet
from duibi.Lvit.LVit import LViT,LViT1,LViT2,LViT0
from torchvision import transforms
from duibi.TGANet.tganet import TGAPolypSeg
# from duibi.LVAT.LV import LVAT
from duibi.LVAT.logger import create_logger
import torch.distributed as dist
# from duibi.LVAT.ConVIRt import LSegNet
from duibi.VMUnet.VMunet import VMUNet
# from duibi.EMFormer.EMFormer import *
from MLAT.MLAT import MLAT
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore")
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--train_dataset', type=str,default='/media/share/data6/2511110091/Datasets/AMDSD/train', )
parser.add_argument('--val_dataset', type=str,default='/media/share/data6/2511110091/Datasets/AMDSD/test', )
parser.add_argument('--batch_size', default=4,type=int,help='batchsize')
parser.add_argument('--workers', default=4,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, )
parser.add_argument('--warm_epoch', '-w', default=0, type=int, )
parser.add_argument('--end_epoch', '-e', default=100, type=int, )
parser.add_argument('--num_class', '-t', default=2, type=int,)
parser.add_argument('--device', default='cuda', type=str, )
parser.add_argument('--checkpoint', type=str, default='/media/share/data6/2511110091/MTGT/result/AMDSD', )
parser.add_argument('--save_name', type=str, default= '/0', )
parser.add_argument('--devicenum', default='3', type=str, )
parser.add_argument('--name', default='1', type=str, )
parser.add_argument('--seed', default=2022, type=int, )
parser.add_argument(
    "--config", metavar="FILE", help="Path to a pretraining config file."
)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
begin_time = time.time()

set_seed(seed=args.seed)
device = args.device
if not os.path.exists(args.checkpoint):os.mkdir(args.checkpoint)
model_savedir = args.checkpoint + args.save_name.replace('0',args.name) + r'/'#+'lr'+ str(args.lr)+ 'bs'+str(args.batch_size)+'/'
save_name =model_savedir +'ckpt'
print(model_savedir)
if not os.path.exists(model_savedir):os.mkdir(model_savedir)
epochs = args.warm_epoch + args.end_epoch

# train_imgs = [cv2.resize(np.load(i), (args.resize,args.resize))[:,:,::-1] for i in train_imgs]
train_text = read_text(config.train_dataset + 'Train_text.xlsx')
val_text = read_text(config.val_dataset + 'Val_text.xlsx')
train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

best_acc_final = []
def main():
    cudnn.benchmark = False
    cudnn.deterministic = True

    config_vit = config.get_CTranS_config()
    model = MLAT(config_vit, n_channels=config.n_channels, n_classes=6)



    # model.encoder.load_state_dict(torch.load('tools_seg/resnet34-333f7ec4.pth'))
    model= model.to('cuda')

    train_dataset = ImageToImage2D(args.train_dataset, config.task_name, train_text, train_tf,
                                   image_size=config.img_size)
    val_dataset = ImageToImage2D(args.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)

    criterion = nn.CrossEntropyLoss(weight=None).to('cuda') #weight=torch.tensor([1,10]

    best_model_wts = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    train_dl = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,pin_memory=False,num_workers=0,drop_last=True,)
    val_dl = DataLoader(val_dataset,batch_size=args.batch_size,pin_memory=False,num_workers=0,)
    best_acc = 0
    with tqdm(total=epochs, ncols=60) as t:
        for epoch in range(epochs):
            epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou = \
                fit(epoch,epochs,model,train_dl,val_dl,device,criterion,optimizer,CosineLR)

            f = open(model_savedir + 'log'+'.txt', "a")
            f.write('epoch' + str(float(epoch)) +
                    '  _train_loss'+ str(epoch_loss)+'  _val_loss'+str(epoch_val_loss)+
                    ' _epoch_acc'+str(epoch_iou)+' _val_iou'+str(epoch_val_iou)+   '\n')

            if epoch_val_iou > best_acc:
                f.write( '\n' + 'here' + '\n')
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_val_iou
                torch.save(best_model_wts, ''.join([save_name,  '.pth']))
            torch.save(best_model_wts, ''.join([save_name, 'last.pth']))
            f.close()
            # torch.cuda.empty_cache()
            t.update(1)
    write_options(model_savedir,args,best_acc)


    # dice,pre,recall,f1_score ,pa = test_mertric_here(model,test_imgs,test_masks,save_name)
    # f = open('./checkpoint/result_txt/' + 'r34u_base_train'+'.txt', "a")
    # f.write(str(model_savedir)+'  dice'+str(dice)+'  pre'+str(pre)+'  recall'+str(recall)+
    #         '  f1_score'+str(f1_score)+'  pa'+str(pa)+'\n')
    # f.close()
    # print('test_acc',acc)
    # print('best_acc','%.4f'%best_acc)

if __name__ == '__main__':
    main()


# print(save_name)