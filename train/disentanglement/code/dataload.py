# -*- coding: utf-8 -*-


import os
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import librosa
import time
import copy



class GET_MFCC(data.Dataset):
    def __init__(self,
                 dataset_dir,phase):



        self.data_path = dataset_dir
        self.phase = phase

        self.emo_number = [0,1,2,3,4,5,6,7]

        if phase == 'test':
            self.con_number = [i for i in range(0.15*len(os.pathdir(dataset_dir+'0/')))]
        elif phase == 'train':
            self.con_number = [i for i in range(0.15*len(os.pathdir(dataset_dir+'0/')),len(os.pathdir(dataset_dir+'0/')))]



    def __getitem__(self, index):
        #select name
        '''
        idx1, idx2,idx3, idx4 = np.random.choice(4,size=4)
        nidx1, nidx2,nidx3, nidx4 = self.name[idx1],self.name[idx2],self.name[idx3],self.name[idx4]

        idx1, idx2 = np.random.choice(4,size=2)
        oidx1, oidx2 = self.name[idx1],self.name[idx2]
        '''
        # select two emotions
        idx1, idx2 = np.random.choice(len(self.emo_number), size=2, replace=False)
        eidx1, eidx2 = self.emo_number[idx1], self.emo_number[idx2]
        # select two contents
        idx1, idx2, idx3 = np.random.choice(len(self.con_number), size=3, replace=True)

        cidx1, cidx2, cidx3= self.con_number[idx1],self.con_number[idx2], self.con_number[idx3]

        audio_path11 = os.path.join(self.data_path,str(eidx1)+'/'+str(cidx1)+'.pkl' )
  #      audio_path22 = os.path.join(self.data_path,str(eidx2)+'/'+str(cidx2)+'.pkl' )
        audio_path12 = os.path.join(self.data_path,str(eidx2)+'/'+str(cidx1)+'.pkl' )
  #      audio_path21 = os.path.join(self.data_path,str(eidx1)+'/'+str(cidx2)+'.pkl' )
        audio_path21 = os.path.join(self.data_path,str(eidx1)+'/'+str(cidx2)+'.pkl' )
        audio_path32 = os.path.join(self.data_path,str(eidx2)+'/'+str(cidx3)+'.pkl' )



        f=open(audio_path11,'rb')
        mfcc11=pickle.load(f)
        mfcc11 = torch.FloatTensor(mfcc11[:,1:])
        f.close()

#        f=open(audio_path22,'rb')
#        mfcc22=pickle.load(f)
#        mfcc22 = torch.FloatTensor(mfcc22[:,:12])
#        f.close()

        f=open(audio_path12,'rb')
        mfcc12=pickle.load(f)
        mfcc12 = torch.FloatTensor(mfcc12[:,1:])
        f.close()

        f=open(audio_path21,'rb')
        mfcc21=pickle.load(f)
        mfcc21 = torch.FloatTensor(mfcc21[:,1:])
        f.close()

        f=open(audio_path32,'rb')
        mfcc32=pickle.load(f)
        mfcc32 = torch.FloatTensor(mfcc32[:,1:])
        f.close()


        mfcc11=torch.unsqueeze(mfcc11, 0).cuda()
        mfcc21=torch.unsqueeze(mfcc21, 0).cuda()
        mfcc12=torch.unsqueeze(mfcc12, 0).cuda()
        mfcc32=torch.unsqueeze(mfcc32, 0).cuda()

        target11 = mfcc11.detach().clone()
#        target22 = mfcc22.detach().clone()

        target12 = mfcc12.detach().clone()
#        target21 = mfcc21.detach().clone()

        label1 = torch.tensor(eidx1).long().cuda()
        label2 = torch.tensor(eidx2).long().cuda()

        return {"input11": mfcc11, "target11": target11,
               "target21": target11, "target22": target12,
                "input12": mfcc12, "target12": target12,
                "label1": label1,  "label2": label2,
                "input21": mfcc21, "input32": mfcc32
              }


    def __len__(self):

       # return self.all_number * len(self.emo_number)
        return len(self.con_number) * len(self.emo_number)