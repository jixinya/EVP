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

MEAD = {'angry':0, 'contempt':1, 'disgusted':2, 'fear':3, 'happy':4, 'neutral':5,
        'sad':6, 'surprised':7}

class SER_MFCC(data.Dataset):
    def __init__(self,
                 dataset_dir,train):

      #  self.data_path = dataset_dir
      #  file = open('/media/asus/840C73C4A631CC36/MEAD/SER_new/list.pkl', "rb") #'rb'-read binary file
      #  self.train_data = pickle.load(file)
      #  file.close()

        self.data_path = dataset_dir

        self.train = train
        if(self.train=='train'):
            file = open('../train_M030.pkl', "rb") #'rb'-read binary file
            self.train_data = pickle.load(file)
            file.close()
        if(self.train=='val'):
            file = open('../val_M030.pkl', "rb") #'rb'-read binary file
            self.train_data = pickle.load(file)
            file.close()



    def __getitem__(self, index):


        emotion = self.train_data[index].split('_')[0]

        label = torch.Tensor([MEAD[emotion]])

        mfcc_path = os.path.join(self.data_path ,  self.train_data[index])


        file = open(mfcc_path,'rb')
        mfcc = pickle.load(file)
        mfcc = mfcc[:,1:]
        mfcc = torch.FloatTensor(mfcc)
        mfcc=torch.unsqueeze(mfcc, 0)
        file.close()


        return mfcc,label


    def __len__(self):

        return len(self.train_data)
