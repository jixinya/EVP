# -*- coding: utf-8 -*-


import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import numpy as np
from collections import OrderedDict
import argparse
import pickle
import torch.nn.functional as F
from dataload import SER_MFCC

from model import EmotionNet,DisNet

from torch.nn import init
from sklearn.model_selection import train_test_split
import tensorboardX

def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=100)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="train/MFCC/M030/")

    parser.add_argument("--model_dir",
                        type=str,
                        default="train/emotion_pretrain/model_M030/")

    parser.add_argument('--device_ids', type=str, default='0')


    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')


    return parser.parse_args()

config = parse_args()

os.makedirs(config.model_dir, exist_ok = True)

###########load model#######################
torch.backends.cudnn.benchmark = True
model = EmotionNet()

CroEn_loss =  nn.CrossEntropyLoss()
tripletloss = nn.TripletMarginLoss(margin=1)
if config.cuda:
    device_ids = [int(i) for i in config.device_ids.split(',')]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    CroEn_loss = CroEn_loss.cuda()
    tripletloss = tripletloss.cuda()

initialize_weights(model)
opt_m = torch.optim.Adam(model.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))


def compute_acc(input_label, out):
    _, pred = out.topk(1, 1)
    pred0 = pred.squeeze().data
    acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
    return acc


###########load data#######################
#dataset = SER_MFCC(config.dataset_dir)
print('load data begin')
train_set = SER_MFCC(config.dataset_dir,'train')
val_set = SER_MFCC(config.dataset_dir,'val')
#train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=1)
train_loader = DataLoader(train_set,batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
val_loader = DataLoader(val_set,batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
print('load data end')
###########train#######################
writer = tensorboardX.SummaryWriter(comment='M030')

total_steps = 0
train_iter = 0
val_iter = 0

a = time.time()
for epoch in range(config.start_epoch, config.max_epochs):
    epoch_start_time = time.time()

    all_acc = 0.0
 #   data_time = time.time()
    for i, (data, label) in enumerate(train_loader):
        iter_start_time = time.time()
        train_iter += 1

        data = Variable(data.float().cuda())
        label = Variable(label.long().cuda())

        label=torch.squeeze(label)


        fake = model(data)
        loss = CroEn_loss(fake,label)
        acc  =  compute_acc(label, fake)

        writer.add_scalar('Trian',loss,train_iter)

        opt_m.zero_grad()
        loss.backward()
        opt_m.step()


        all_acc += acc.item()

        if (train_iter % 1000 == 0):

            print('[%d,%5d / %d] train loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(train_loader), loss.item(),time.time()-a))

    writer.add_scalar('Train_acc',float(all_acc)/(i+1),epoch+1)

    string = os.path.join(config.model_dir,'SER_'+str(epoch) + '.pkl')
    # save model parameters
    torch.save(model.state_dict(), string)

    model.eval()

    all_val_acc = 0.0
    print("start to validate, epoch %d" %(epoch+1))
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):

            val_iter += 1

            data = Variable(data.float().cuda())
            label = Variable(label.long().cuda())

            label = torch.squeeze(label)
            fake = model(data)


            loss_t = CroEn_loss(fake,label)
            val_acc  =  compute_acc(label, fake)


            writer.add_scalar('Val',loss_t,val_iter)



            all_val_acc += val_acc.item()
            if (val_iter % 1000 == 0):

                print('[%d,%5d / %d] test loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(val_loader), loss_t.item(),time.time()-a))
    writer.add_scalar('Val_acc',float(all_val_acc)/(i+1), epoch+1)
