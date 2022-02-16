#!/usr/bin/env python3
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

from dataload import GET_MFCC

from models_content_cla import AutoEncoder2x

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

def initialize_weights( net, init_type='kaiming', gain=0.02):
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
                        #default="/media/asus/840C73C4A631CC36/MEAD/dataset/emotion_same_length/")
                        default="train/disentanglement/emotion_length/M030/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_dir",
                        type=str,
                        default="train/disentanglement/model_M030/")
                        # default="/mnt/disk1/dat/lchen63/grid/model/model_gan_r")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--image_dir",
                        type=str,
                        default="train/disentanglement/image_M030/")
    parser.add_argument("--log_dir",
                        type=str,
                        default="train/disentanglement/log_M030/")

    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--triplet_margin', type=int, default=1)
    parser.add_argument('--triplet_weight', type=int, default=10)
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_dir', type=str,default='/media/thea/Data/New_exp/3_intensity_M030/SER_intensity_3/model/81_pretrain.pth')
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--use_triplet', type=bool, default=False)
    parser.add_argument('--atpretrained_dir', type=str,default='train/disentanglement/atnet_lstm_18.pth')
    parser.add_argument('--serpretrained_dir', type=str,default='train/emotion_pretrain/model_M030/SER_99.pkl')
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--rnn', type=bool, default=True)

    return parser.parse_args()

config = parse_args()
os.makedirs(config.model_dir,exist_ok = True)
###########load model#######################
torch.backends.cudnn.benchmark = True
model = AutoEncoder2x(config)
#CroEn_loss =  nn.CrossEntropyLoss()
if config.cuda:
    device_ids = [int(i) for i in config.device_ids.split(',')]
    model = model.cuda()

#    CroEn_loss = CroEn_loss.cuda()

initialize_weights(model)

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


if config.pretrain :
    # ATnet resume
    pretrain = torch.load(config.atpretrained_dir)
    tgt_state = model.state_dict()
    strip = 'module.'
    for name, param in pretrain.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
            name = 'con_encoder.'+name
        if name not in tgt_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        tgt_state[name].copy_(param)
        print(name)

    #SER resume
    pretrain = torch.load(config.serpretrained_dir)
  #  tgt_state = model.state_dict()

    strip = 'module.'
    for name, param in pretrain.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
            name = 'emo_encoder.'+name
        if name not in tgt_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        if name in tgt_state:

            tgt_state[name].copy_(param)
            print(name)
    strip = 'module.'
    for name, param in pretrain.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
            name = 'classify.'+name
        if name not in tgt_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        if name in tgt_state:

            tgt_state[name].copy_(param)
            print(name)





###########load data#######################
#dataset = GET_MFCC(config.dataset_dir)
print('start split')
train_set = GET_MFCC(config.dataset_dir,'train')
test_set = GET_MFCC(config.dataset_dir,'test')
#train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=1)
#train_set = GET_MFCC(config.dataset_dir,'train')
#test_set = GET_MFCC(config.dataset_dir,'test')
#train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=1)
train_loader = DataLoader(train_set,batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
test_loader = DataLoader(test_set,batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
print('end split')
###########train#######################
writer = tensorboardX.SummaryWriter(comment='M030')

total_steps = 0
train_iter = 0
val_iter = 0

start_epoch = config.start_epoch
if config.resume :
    # ATnet resume
    resume = torch.load(config.resume_dir)
    tgt_state = model.state_dict()
    train_iter = resume['train_step']
    val_iter = resume['test_step']
    start_epoch = resume['epoch']
    resume_state = resume['model']
    model.load_state_dict(resume_state)
    print('load resume model')

a = time.time()
for epoch in range(start_epoch, config.max_epochs):
    epoch_start_time = time.time()

    acc_1 = 0.0
    acc_2 = 0.0

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()


     #   data = Variable(data.float().cuda())

        outputs, losses, acces = model.train_func(data)

        losses_values = {k:v.item() for k, v in losses.items()}
        acces_values = {k:v.item() for k, v in acces.items()}

        # record loss to tensorboard
        for k, v in losses_values.items():
            writer.add_scalar(k, v, train_iter)

        acc_1 += acces_values['acc_1']
        acc_2 += acces_values['acc_2']


        loss = sum(losses.values())
        writer.add_scalar('train_loss', loss, train_iter)

      #  opt_m.zero_grad()



      #  loss.backward()

      #  opt_m.step()

        if (train_iter % 10 == 0):

            #for k, v in losses_values.items():
            #    print(k,v)

            print('[%d,%5d / %d] con_feature loss :%.10f cla_1 loss :%.10f train loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(train_loader), losses['con_feature'].item(),losses['cla_1'].item(),loss.item(),time.time()-a))

        if (train_iter % 100 ==0): #500
            with open(config.log_dir + 'train.txt','a') as file_handle:
                file_handle.write('[%d,%5d / %d] con_feature loss :%.10f cla_1 loss :%.10f train loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(train_loader), losses['con_feature'].item(),losses['cla_1'].item(),loss.item(),time.time()-a))
                file_handle.write('\n')

        if (train_iter % 500 == 0): #2000


            save_path = os.path.join(config.image_dir+'train/'+str(epoch+1),str(train_iter))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_fig(data,outputs,save_path)

        train_iter += 1
#    string = os.path.join(config.model_dir,'SER_'+str(epoch) + '.pkl')
    # save model parameters
#    torch.save(model.state_dict(), string)
    writer.add_scalar('acc_1',float(acc_1)/(i+1),epoch+1)
    writer.add_scalar('acc_2',float(acc_2)/(i+1),epoch+1)

    torch.save({
                'train_step': train_iter,
                'test_step': val_iter,
                'epoch': epoch,
                'model': model.state_dict(),
                 }, os.path.join(config.model_dir, str(epoch) + "_"  + 'pretrain.pth'))



   # model.eval()


    print("start to validate, epoch %d" %(epoch+1))

    acc_1_v = 0.0
    acc_2_v = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader):



            outputs, losses, acces = model.val_func(data)

            losses_values = {k:v.item() for k, v in losses.items()}
            acces_values = {k:v.item() for k, v in acces.items()}

        # record loss to tensorboard
            for k, v in losses_values.items():
                writer.add_scalar(k+'_v', v, val_iter)

            acc_1_v += acces_values['acc_1']
            acc_2_v += acces_values['acc_2']

            loss = sum(losses.values())
            writer.add_scalar('test_loss', loss, val_iter)

            if (val_iter % 10 == 0):

                print('[%d,%5d / %d] con_feature loss :%.10f cla_1 loss :%.10f val loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(test_loader), losses['con_feature'].item(),losses['cla_1'].item(),loss.item(),time.time()-a))

            if (val_iter % 500 == 0): #2000


                save_path = os.path.join(config.image_dir+'val/'+str(epoch+1),str(val_iter))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model.save_fig(data,outputs,save_path)

                with open(config.log_dir + 'val.txt','a') as file_handle:
                    file_handle.write('[%d,%5d / %d] con_feature loss :%.10f cla_1 loss :%.10f val loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(test_loader), losses['con_feature'].item(),losses['cla_1'].item(),loss.item(),time.time()-a))
                    file_handle.write('\n')

            val_iter += 1
        writer.add_scalar('acc_1_v',float(acc_1_v)/ (i+1), epoch+1)
        writer.add_scalar('acc_2_v',float(acc_2_v)/ (i+1), epoch+1)

