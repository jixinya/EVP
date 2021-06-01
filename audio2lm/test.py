
import sys
sys.path.append('./')


import argparse
import os
import glob
import time
import yaml
import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import librosa
from models import AT_emoiton
import cv2
#import scipy.misc
from utils import VideoWriter, draw_mouth, add_audio, check_volume, change_mouth
#from tqdm import tqdm
#import torchvision.transforms as transforms
import shutil
from collections import OrderedDict
import python_speech_features

#from scipy.spatial import procrustes
import matplotlib.pyplot as plt



#import filter1

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",
                     type=int,
                     default=1)
    parser.add_argument("--cuda",
                     default=True)
    parser.add_argument("--lstm",
                     default=True)
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
    parser.add_argument("--config", required=True , help="path to config")#default='config/M003_test.yaml'
    parser.add_argument("--condition", default='feature', help="audio/feature, choose which to use, recommand feature")
    parser.add_argument('-i','--audio', type=str, default='audio/obama.wav')#'/home/thea/data/MEAD/ATnet_emotion/audio/M003_72_1_output_03.wav'
    parser.add_argument('--emo_audio', type=str, default='audio/M003_happy.wav')#/home/thea/data/MEAD/MEAD_demo/M03_crop/M030_01_3_output_01.wav
    parser.add_argument('--emo_feature', type=str, default='M003_feature/angry.npy')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=1)   
    return parser.parse_args()


def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
            
        return new_state_dict
    else:
        return state_dict


def test(opt, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_ids
 
    
    pca = torch.FloatTensor( np.load(config['pca'])[:,:16]).cuda()#20
    mean =torch.FloatTensor( np.load(config['mean'])).cuda()
    
  
    encoder = AT_emoiton(opt, config)
    if opt.cuda:
        encoder = encoder.cuda()

    state_dict = multi2single(config['at_model'], 0)
    encoder.load_state_dict(state_dict)

    encoder.eval()

    test_file = opt.audio
    emo_file = opt.emo_audio
    
    example_landmark = np.load(config['mean']) #'../basics/mouth_close.npy'
    example_landmark=example_landmark.reshape(106,2)  #150*2 

    example_landmark =  example_landmark.reshape((1,example_landmark.shape[0]* example_landmark.shape[1])) #1.300

    if opt.cuda:

        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)) ).cuda()

    
    # Load speech and extract features

    example_landmark  = example_landmark - mean.expand_as(example_landmark)
    example_landmark = torch.mm(example_landmark,  pca) 
    
    speech, sr = librosa.load(test_file, sr=16000)
    clip = check_volume(speech,sr)
    
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)
    
    print ('=======================================')
    print ('Start to generate images')

    ind = 3
    with torch.no_grad(): 
        fake_lmark = []
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)
        
        if opt.condition == 'audio':
            speech, sr = librosa.load(emo_file, sr=16000)

            speech = np.insert(speech, 0, np.zeros(1920))
            speech = np.append(speech, np.zeros(1920))
            mfcc_emo = python_speech_features.mfcc(speech,16000,winstep=0.01)
            ind = 3
            emo_mfcc = []
            while ind <= int(mfcc_emo.shape[0]/4) - 4:
                t_mfcc =mfcc_emo[( ind - 3)*4: (ind + 4)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc).cuda()
                emo_mfcc.append(t_mfcc)
                ind += 1
            emo_mfcc = torch.stack(emo_mfcc,dim = 0)
        
            if(emo_mfcc.size(0) > input_mfcc.size(0)):
                emo_mfcc = emo_mfcc[:input_mfcc.size(0),:,:]
            
            if(emo_mfcc.size(0) < input_mfcc.size(0)):
                n = input_mfcc.size(0) - emo_mfcc.size(0)
                add = emo_mfcc[-1,:,:].unsqueeze(0)
                for i in range(n):
                    emo_mfcc = torch.cat([emo_mfcc,add],0)
        
            input_mfcc = input_mfcc.unsqueeze(0)
            emo_mfcc = emo_mfcc.unsqueeze(0)
            fake_lmark = encoder(example_landmark, input_mfcc,emo_mfcc)
            fake_lmark = fake_lmark.view(fake_lmark.size(0) *fake_lmark.size(1) , 16)  
        elif opt.condition == 'feature':
            
            emo_f = []
            emo_feature = np.load(opt.emo_feature)
            t_mfcc = torch.FloatTensor(emo_feature).cuda()
            for i in range(input_mfcc.size(0)):
                emo_f.append(t_mfcc)
            emo_f = torch.stack(emo_f,dim = 0)
            input_mfcc = input_mfcc.unsqueeze(0)
            emo_f = emo_f.unsqueeze(0)
    
            fake_lmark = encoder.feature_input(example_landmark, input_mfcc,emo_f)
            fake_lmark = fake_lmark.view(fake_lmark.size(0) *fake_lmark.size(1) , 16) 
        else:
            raise Exception('condition wrong: can only be audio/feature.')
        
        
      
      #if get D-value
        fake_lmark=fake_lmark + example_landmark.expand_as(fake_lmark)

        fake_lmark = torch.mm( fake_lmark, pca.t() )
        fake_lmark = fake_lmark + mean.expand_as(fake_lmark)

        fake_lmark = fake_lmark.unsqueeze(0) 
            
        fake_lmark = fake_lmark.data.cpu().numpy()
        
        fake_lmark = np.reshape(fake_lmark, (fake_lmark.shape[1], 106,2))
        
        fake_lmark = change_mouth(fake_lmark, clip)
        
        np.save(config['sample_dir'], fake_lmark)
        
        mouth_video_writer = VideoWriter(config['video_dir'], 256, 256, 25)
    
   
        mouth_img = []
        for i in range(len(fake_lmark)):
            mouth_img.append(draw_mouth(fake_lmark[i]*255, 256, 256))
            mouth_video_writer.write_frame(draw_mouth(fake_lmark[i]*255, 256, 256))
        mouth_video_writer.end()

        add_audio(config['video_dir'], opt.audio)
 
        print ('The generated video is: {}'.format(config['video_dir'].replace('.mp4','.mov')))

if __name__ == "__main__":
    opt = parse_args()        
    with open(opt.config) as f:
        config = yaml.load(f)
    test(opt,config)

