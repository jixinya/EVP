#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:59:19 2019

@author: thea
"""


import time
import argparse
import os
import glob

import numpy as np

import librosa
import utils
import re
import python_speech_features
import cv2
import scipy.misc
from tqdm import tqdm

import shutil
from collections import OrderedDict
import python_speech_features
from skimage import transform as tf
from copy import deepcopy
from scipy.spatial import procrustes

import dlib
import pickle

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg
import utils

ABSOLUTELY_00001.npy
ABUSE_00001.npy
xLim=(0, 1.0)
yLim=(0, 1.0)
xLab = 'x'
yLab = 'y'
filepath='/home/thea/data/jxy/ATVG-test/first-2D frontalize/re/'
pathDir = os.listdir(filepath)

font = {'size'   : 18}

def plot_flmarks(pts,b, lab, xLim, yLim, xLab, yLab, figsize=(10, 10)):
    if len(pts.shape) != 2:
        pts = np.reshape(pts, (pts.shape[0]/2, 2))
    
    
    plt.figure(figsize=figsize)
    plt.plot(pts[:,0], pts[:,1], 'ko',color='r', ms=4)
    plt.plot(b[:,0], b[:,1], 'ko',color='b', ms=4)
#    for refpts in lookup:
#        plt.plot([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]], 'k', ms=4)
    
    plt.xlabel(xLab, fontsize = font['size'] + 4, fontweight='bold')
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top') 
    plt.ylabel(yLab, fontsize = font['size'] + 4, fontweight='bold')
    plt.xlim(xLim)
    plt.ylim(yLim)
    plt.gca().invert_yaxis()
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

name = 'M003_01_3_output_01'

def getraw(name):
    a=np.load('/home/thea/data/MEAD/ATnet_emotion/dataset/106_landmark/'+name+'/0.npy')
    for i in range(len(a)):
        b=a[i].reshape(68,2)
        path='/home/thea/data/MEAD/ATnet_emotion/temp/img'
      #  path=os.path.join('/home/thea/data/MEAD/ATnet_emotion/pca/68/raw',name)
        if not os.path.exists(path):
            os.mkdir(path)
        lab="{}/{:05d}.png".format(path ,i)
#        lab=os.path.join(path,str(i)+'.jpg')
        plot_flmarks(b, lab,(0,1.0),(0,1.0),'x','y', figsize=(10, 10))
        print('Generate %d jpg' % i)
        
getraw('WEEKEND_00768')
#DAVID_00560
#GROWTH_00020
#LEAST_00578
#WEEKEND_00768
#INVOLVED_00575
def getpca(name,n):
    a=np.load('/home/thea/data/MEAD/ATnet_emotion/dataset/106_landmark/'+name+'/0.npy')
    pca = np.load('/home/thea/data/MEAD/ATnet_emotion//basics/U_106.npy')[:,:20] 
    mean= np.load('/home/thea/data/MEAD/ATnet_emotion/basics/mean_106.npy')
    for i in range(len(a)):
        b=a[i].reshape(212,)
        lmark = b - mean#expand as the size of lmark
        lmark=np.dot(lmark,pca)
        result=np.dot(lmark,pca.T)+mean
        result=result.reshape(106,2)
        b=b.reshape(106,2)
        path='/home/thea/data/MEAD/ATnet_emotion/pca/106/'+'pca-'+str(20)+'/'+name
        if not os.path.exists(path):
            os.mkdir(path)
        lab=os.path.join(path,str(i)+'.jpg')
        plot_flmarks(result, b,lab,(0,1.0),(0,1.0),'x','y', figsize=(10, 10))
        print('Generate %d jpg' % i)
getpca('WEEKEND_00768',6)

c=[3,15,20,25,30,40,50]
for i in range(len(c)):
    getpca('DAVID_00560',c[i])


def getall(n):
    filepath='/home/thea/data/jxy/ATVG-test/first-2D frontalize/'+'pca-'+str(n)
    pathDir = os.listdir(filepath)
    for i in range(len(pathDir)):
        allpath=os.path.join(filepath,pathDir[i])
        count=0
        for j in range(29):
            imagepath=os.path.join(allpath,str(j)+'.jpg')
            img = cv2.imread(imagepath)
            k=count+i*29
            path=filepath+'/'+str(n)+'/'
            if not os.path.exists(path):
                os.mkdir(path)
            savepath=path+str(k)+'.jpg'
            cv2.imwrite(savepath, img)
            count+=1

for i in range(len(c)):
    getall(c[i])


filepath='/home/thea/data/jxy/ATVG-test/first-2D frontalize/raw_image/'
pathDir = os.listdir(filepath)
for i in range(len(pathDir)):
    allpath=os.path.join(filepath,pathDir[i])
    count=0
    for n in range(29):
        imagepath=os.path.join(allpath,str(n)+'.jpg')
        img = cv2.imread(imagepath)
        j=count+i*29
        savepath='/home/thea/data/jxy/ATVG-test/first-2D frontalize/raw_image/1/'+str(j)+'.jpg'
        cv2.imwrite(savepath, img)
        count+=1
        
filepath='/media/thea/其他/pca/overlap/video/'
pathDir = os.listdir(filepath)
count=0
for i in range(len(pathDir)):
    child=os.path.join(filepath,pathDir[i])
#child='/media/thea/其他/pca/overlap/video/'       
    cap = cv2.VideoCapture(child)
#count=0
#        success = True
      
    while True:
        success,image = cap.read()
        if success:
            cv2.imwrite("/media/thea/其他/pca/overlap/image/%d.jpg" % count, image)
            print(count)
            count+=1
        
        else:
            break        

#overlap
filepath='/home/thea/data/jxy/ATVG-test/test-set/process'
pathDir = os.listdir(filepath)

s=0
f=0
for i in range(len(pathDir)):#len(pathDir)
    allpath=os.path.join(filepath,pathDir[i]+'.txt')
    
    count=0
    lmark_all=[]
    for a in open(allpath,'r'):
        count+=1
        a=a[:-1]
        a=a.split(' ')
        a=a[:-1]
        if(count==int(a[0])):
            amark=np.zeros((300,1))
            for k in range(300):
                amark[k]=float(a[k+1])
            lmark_all.append(amark)
    if(len(lmark_all)==29):
        landmark_path = os.path.join('/home/thea/data/Net/3DMM',pathDir[i].replace('txt', 'npy'))
        np.save(landmark_path,lmark_all)
        s+=1
        print('Generate %d txt success.There are in total %d txt' % (s,i))
    else:
        f+=1
        print('Generate %d txt fail.There are in total %d txt' % (f,i))

filepath='/home/thea/data/Net/3DMM'
pathDir = os.listdir(filepath)
apoints=np.zeros((145,340))
count=0
for i in range(len(pathDir)):
    allpath=os.path.join(filepath,pathDir[i])
    a=np.load(allpath).reshape(29,340)
    for j in range(len(a)):
        apoints[count]=a[j]
        count+=1

#Generate nlmarks
filepath='/media/thea/其他/pca/overlap/npy/'
pathDir = os.listdir(filepath)
path='/media/thea/其他/prepocess/nlmark_v'
points=np.zeros((145,340))
count=0
for i in range(len(pathDir)):
    allpath=os.path.join(path,pathDir[i])
    a=np.load(allpath)
    for j in range(len(a)):
        points[count]=a[j]
        count+=1
    
    
#Generate pictures
path='/media/thea/其他/pca/overlap/norm/'
for i in range(len(apoints)):
    b=points[i].reshape(170,2)
    lab=os.path.join(path,str(i)+'.jpg')
    utils.plot_flmarks(b, lab, xLim, yLim, xLab, yLab, figsize=(10, 10))
    print('Generate %d jpg' % i)


#test points
filepath='/home/thea/data/MEAD/lamrk2image/LRW_image/GREAT_00004/'
#path='/media/thea/其他/pca/overlap/image/'
pathDir = os.listdir(filepath)
count=0
b=np.load('/home/thea/data/MEAD/lamrk2image/LRW_lm/GREAT_00004.npy')
for i in range(len(pathDir)):
    path=os.path.join(filepath,str(i)+'.jpg')
    
        
        
    
   
    a=b[i]
    a=a.reshape(106,2)
#   
    img = cv2.imread(path, cv2.IMREAD_COLOR) 
    for k in range(len(a)):
            
        tup=(int(a[k,0]),int(a[k,1]))
        cv2.circle(img, tup, 0, (0,0, 255), 2)
 #   
    cv2.imwrite(os.path.join('/home/thea/data/MEAD/lamrk2image/check',str(i)+'.jpg'),img)
  #  cv2.imshow('image',img)
  #  cv2.waitKey (10000) # 显示 10000 ms 即 10s 后消失
  #  cv2.destroyAllWindows()
    print(i)
    
    
path = '/media/thea/其他/pca/overlap/result/'
filelist = os.listdir(path)

fps = 25 #视频每秒24帧
size = (256, 256) #需要转为视频的图片的尺寸
#可以使用cv2.resize()进行修改

video = cv2.VideoWriter("/media/thea/其他/pca/overlap/Video.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
#视频保存在当前目录下

for i in range(len(filelist)):
    
    #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
    item = path + str(i)+'.jpg'
    img = cv2.imread(item)
    video.write(img)

video.release()
cv2.destroyAllWindows()

#pca normalization
filepath='/home/thea/data/jxy/ATVG-test/test-set/nlmark'
pathDir = os.listdir(filepath)
pca_all=[]

for i in range(len(pathDir)):
    path=os.path.join(filepath,pathDir[i])
    a=np.load(path)
   
    pca = np.load('/home/thea/data/jxy/ATVG-test/test-set/U_lrw1.npy')[:,:20] 
    mean= np.load('/home/thea/data/jxy/ATVG-test/test-set/mean_lrw1.npy')
    for j in range(len(a)):
        b=a[j]
        lmark = b - mean#expand as the size of lmark
        lmark=np.dot(lmark,pca)
        pca_all.append(lmark)
    print('Generate %d pca' % i)

pca_a=np.array(pca_all)
mean=np.mean(pca_a,axis=0)
gap=np.zeros((20,1))
for i in range(20):
    a=pca_a[:,i]
    b=np.max(a)-np.min(a)
    gap[i]=b

np.save('/home/thea/data/jxy/ATVG-test/test-set/pca_mean.npy',mean)
np.save('/home/thea/data/jxy/ATVG-test/test-set/pca_gap.npy',gap)


#check pca
xLim=(-1.0, 1.0)
yLim=(-1.0, 1.0)
xLab = 'x'
yLab = 'y'
def check(m,n):
    pca = np.load('/home/thea/data/Net/U_lrw1.npy')[:,:6] 
    mean= np.load('/home/thea/data/Net/mean_lrw1.npy')
    a=np.zeros((1,6))
    a[0,m]=n
    result=np.dot(a,pca.T)+mean
    result=result.reshape(68,2)
    path=os.path.join('/home/thea/data/Net/6pca/',str(m))
    if not os.path.exists(path):
        os.mkdir(path)
    lab=os.path.join(path,str(n)+'.jpg')
    
    utils.plot_flmarks(result, lab, xLim, yLim, xLab, yLab, figsize=(10, 10))

for m in range(0,6):
    check(m,0.5)
    check(m,-0.5)
    check(m,1)
    check(m,-1)
    print(m)

r=np.load('/home/thea/data/MEAD/ATnet_emotion/basics/mean_106.npy')
pca = np.load('/home/thea/data/MEAD/ATnet_emotion//basics/U_106.npy')[:,:20] 
mean= np.load('/home/thea/data/MEAD/ATnet_emotion//basics/mean_106.npy')
lmark = r - mean#expand as the size of lmark
lmark=np.dot(lmark,pca)
result=np.dot(lmark,pca.T)+mean
result=result.reshape(106,2)
r=r.reshape(106,2)
plt.plot(r[:,0], 1-r[:,1], 'ko', color='r',ms=4)
plt.plot(result[:,0], 1-result[:,1], 'ko',color='b', ms=4)



xLim = (0,1.0)
yLim = (0,1.0)
xLab = 'x' 
yLab = 'y'
plt.figure(figsize=(10, 10))
   
plt.plot(r[:,0], r[:,1], 'ko', ms=4)
#    for refpts in lookup:
#        plt.plot([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]], 'k', ms=4)
    
plt.xlabel(xLab, fontsize = font['size'] + 4, fontweight='bold')
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top') 
plt.ylabel(yLab, fontsize = font['size'] + 4, fontweight='bold')
plt.xlim(xLim)
plt.ylim(yLim)
plt.gca().invert_yaxis()
plt.savefig('/home/thea/data/MEAD/ATnet_emotion/results/5.5_dataset/mmean_106.png', dpi = 300, bbox_inches='tight')
plt.clf()
plt.close()
