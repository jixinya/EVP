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
#from visual_loss import Visualizer
#from torchnet import meter

from dataset_difference import  SMED_1D_lstm_landmark_pca
from tensorboardX import SummaryWriter
from models import AT_emoiton
#from visual_loss import Visualizer
from torch.nn import init
import utils
from sklearn.model_selection import train_test_split

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
                        default=16)#192,96
    parser.add_argument("--max_epochs",
                        type=int,
                        default=100)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../dataset_M003/")
                        # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_dir",
                        type=str,
                        #default="../model_M003_mouth_close/")
                        default="../model_M003/")
    parser.add_argument('--pretrained_dir',
                        type=str,
                        #default='/media/asus/840C73C4A631CC36/MEAD/ATnet_emotion/pretrain/M003/90_pretrain.pth'
                        default='train/disentanglement/model_M003/99_pretrain.pth')
    parser.add_argument('--atpretrained_dir', type=str,default='train/disentanglement/atnet_lstm_18.pth')
    parser.add_argument('--serpretrained_dir', type=str,default='train/emotion_pretrain/model_M003/SER_99.pkl')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_sep', type=bool, default=False)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--rnn', type=bool, default=True)

    return parser.parse_args()

config = parse_args()

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

""" 1. load the data """
train_set=SMED_1D_lstm_landmark_pca(config.dataset_dir,'train')
test_set=SMED_1D_lstm_landmark_pca(config.dataset_dir,'test')
#train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=1)
train_loader = DataLoader(train_set,
                         batch_size=config.batch_size,
                         num_workers=config.num_thread,
                         shuffle=True, drop_last=True)

test_loader = DataLoader(test_set,
                         batch_size=config.batch_size,
                         num_workers=config.num_thread,
                         shuffle=True, drop_last=True)


#pca = torch.FloatTensor(np.load('/home/jixinya/first-2D frontalize/U_2Df.npy')[:,:20] ).cuda()#20
#mean = torch.FloatTensor(np.load('/home/jixinya/first-2D frontalize/mean_2Df.npy')).cuda()
num_steps_per_epoch = len(train_loader)
num_val = len(test_loader)

writer=SummaryWriter(comment='M019')

""" 2. load the model """
generator = AT_emoiton(config)
#generator=generator.cuda()
if config.cuda:
    device_ids = [int(i) for i in config.device_ids.split(',')]
    generator   = generator.cuda()
    # self.generator     = self.generator.cuda()

#initialize_weights(generator)


if config.pretrain :
    # ATnet resume
    pretrain = torch.load(config.pretrained_dir)
    pretrain = pretrain['model']
    tgt_state = generator.state_dict()
    strip = 'con_encoder.'
    for name, param in pretrain.items():
        if isinstance(param, nn.Parameter):
            param = param.data
        if strip is not None and name.startswith(strip):
           tgt_state[name].copy_(param)
           print(name)

        if name not in tgt_state:
            continue


    #SER resume

    strip = 'emo_encoder.'
    for name, param in pretrain.items():
        if isinstance(param, nn.Parameter):
            param = param.data
        if name not in tgt_state:
            continue
        if strip is not None and name.startswith(strip):
            tgt_state[name].copy_(param)
            print(name)

if config.pretrain_sep :
    # ATnet resume
    pretrain = torch.load(config.atpretrained_dir)
    tgt_state = generator.state_dict()
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



""" 3. loss function """
#l1_loss_fn   = nn.L1Loss().cuda()
#mse_loss_fn = nn.MSELoss().cuda()

pca =np.load('../basics/U_106.npy')[:, :16]
mean =np.load('../basics/mean_106.npy')


""" 4. train & validation"""
start_epoch = 0

cc = 0
t0 = time.time()
train_itr=0
test_itr=0
for epoch in range(start_epoch, config.max_epochs):


 #           t0 = time.time()
 #           train_loss_meter.reset()
    all_loss = 0.0
    for step, (example_landmark, example_audio, lmark, mfccs) in enumerate(train_loader):
        t1 = time.time()

        if config.cuda:
            lmark    = Variable(lmark.float()).cuda()
            mfccs = Variable(mfccs.float()).cuda()
            example_landmark = Variable(example_landmark.float()).cuda()


        fake_lmark,loss_pca, loss_lm= generator.train_func(  example_landmark, lmark, mfccs)

        loss_pca = 1000*loss_pca
        loss_lm = 1000*loss_lm
        loss = loss_pca + loss_lm
   #     fake = fake_lmark.data.cpu().numpy()
   #     for i in range()
   #     result=np.dot(lmark,pca.T)+mean

        all_loss += loss.item()
        t2=time.time()
        train_itr+=1
      #  writer.add_scalars('Train',{'loss_pca':loss_pca,'loss_lm':loss_lm,'loss':loss},train_itr)
        writer.add_scalar('Train',loss,train_itr)
        writer.add_scalar('Train_lm',loss_lm,train_itr)
        writer.add_scalar('Train_pca',loss_pca,train_itr)



        print("[{}/{}][{}/{}]    loss: {:.8f},data time: {:.4f},  model time: {} second, loss time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch,  loss, t1-t0,  t2 - t1, time.time()-t1))
        print("all time: {} second"
                          .format(time.time() - t1))


    torch.save(generator.state_dict(),
                            "{}/atnet_emotion_{}.pth"
                             .format(config.model_dir,epoch))



    t0 = time.time()
    print("final average train loss = ", float(all_loss)/(step+1))
    #validation


    all_val_loss = 0.0
    for step,  (example_landmark, example_audio, lmark, mfccs) in enumerate(test_loader):
        with torch.no_grad():
            if config.cuda:
                lmark    = Variable(lmark.float()).cuda()
                mfccs = Variable(mfccs.float()).cuda()
                example_landmark = Variable(example_landmark.float()).cuda()


            fake_lmark,loss_pca, loss_lm= generator.val_func( example_landmark, lmark, mfccs)

            loss_pca = 1000*loss_pca
            loss_lm = 1000*loss_lm
            test_loss = loss_pca + loss_lm
          #  test_loss = 1000*test_loss

            all_val_loss += test_loss.item()
            test_itr+=1
            writer.add_scalar('Test',test_loss,test_itr)
            writer.add_scalar('Test_lm',loss_lm,test_itr)
            writer.add_scalar('Test_pca',loss_pca,test_itr)

       #     writer.add_scalars('Val',{'loss_pca':loss_pca,'loss_lm':loss_lm,'loss':test_loss},test_itr)
            print("[{}/{}][{}/{}]   loss: {:.8f},all time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_val, test_loss,   time.time()-t1))

    print("final average test loss = ", float(all_val_loss)/(step+1))












