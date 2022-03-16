# -*- coding: utf-8 -*-


import librosa
import python_speech_features
import numpy as np
import pickle
import librosa.display
import os

from dtw import dtw

from numpy.linalg import norm

import matplotlib.pyplot as plt
import seaborn as sns

sample_interval= 0.01
window_len = 0.025
n_mfcc = 12
sample_delay =14
sample_len = 28

MEAD = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral',
        'sad', 'surprised']

# data process
line_1 =['001','001','028','001','001','001','001','001']
line_2 =['002','002','029','002','002','002','002','002']
line_3 =['003','003','030','003','003','003','003','003']
line_4 =['021','020','018','021','021','031','021','021']
line_5 =['022','021','019','022','022','032','022','022']
line_6 =['023','022','020','023','023','033','023','023']
line_7 =['024','023','021','024','024','034','024','024']
line_8 =['025','024','022','025','025','035','025','025']
line_9 =['026','025','023','026','026','036','026','026']
line_10=['027','026','024','027','027','037','027','027']
line_11=['028','027','025','028','028','038','028','028']
line_12=['029','028','026','029','029','039','029','029']
line_13=['030','029','027','030','030','040','030','030']
line_M003=[]
line_M003.append(line_1)
line_M003.append(line_2)
line_M003.append(line_3)
line_M003.append(line_4)
line_M003.append(line_5)
line_M003.append(line_6)
line_M003.append(line_7)
line_M003.append(line_8)
line_M003.append(line_9)
line_M003.append(line_10)
line_M003.append(line_11)
line_M003.append(line_12)
line_M003.append(line_13)

line_1 =['001','001','001','001','001','001','001','001']
line_2 =['002','002','002','002','002','002','002','002']
line_3 =['003','003','003','003','003','003','003','003']
line_4 =['020','020','021','021','021','031','021','021']
line_5 =['021','021','022','022','022','032','022','022']
line_6 =['022','022','023','023','023','033','023','023']
line_7 =['023','023','024','024','024','034','024','024']
line_8 =['024','024','025','025','025','035','025','025']
line_9 =['025','025','026','026','026','036','026','026']
line_10=['026','026','027','027','027','037','027','027']
line_11=['027','027','028','028','028','038','028','028']
line_12=['028','029','029','029','029','039','029','029']
line_13=['029','030','030','030','030','040','030','030']
line_M030=[]
line_M030.append(line_1)
line_M030.append(line_2)
line_M030.append(line_3)
line_M030.append(line_4)
line_M030.append(line_5)
line_M030.append(line_6)
line_M030.append(line_7)
line_M030.append(line_8)
line_M030.append(line_9)
line_M030.append(line_10)
line_M030.append(line_11)
line_M030.append(line_12)
line_M030.append(line_13)


for i in range(0,13):
    m=line_M030[i]
    con_path='train/disentanglement/Aligned_audio_data/M030/'+str(i)
    if not os.path.exists(con_path):
        os.makedirs(con_path)
    for j in range(0,8):

        if(j == 0):
            audio_path = 'audio2lm/data/M030/audio/'+MEAD[j]+'/'+m[j]+'.m4a'
            y1, sr1 = librosa.load(audio_path,sr=16000)
            y1 = np.insert(y1, 0, np.zeros(1920))
            y1 = np.append(y1, np.zeros(1920))
            mfcc = python_speech_features.mfcc(y1 , sr1 ,winstep=sample_interval)
            with open(con_path+'/'+str(j)+'.pkl', 'wb') as f:
                pickle.dump(mfcc, f)
        else:
            f=open(os.path.join(con_path,'0.pkl'),'rb')
            mfcc1 = pickle.load(f)

            audio_path = 'audio2lm/data/M030/audio/'+MEAD[j]+'/'+m[j]+'.m4a'

            y2, sr2 = librosa.load(audio_path,sr=16000)
            y2 = np.insert(y2, 0, np.zeros(1920))
            y2 = np.append(y2, np.zeros(1920))
            mfcc2 = python_speech_features.mfcc(y2 , sr2 ,winstep=sample_interval)
            dist, cost, acc_cost, path = dtw(mfcc2, mfcc1, dist=lambda x, y: norm(x - y, ord=1))

         #   plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
         #   plt.plot(path[0], path[1], 'w')
         #   plt.xlim((-0.5, cost.shape[0]-0.5))
         #   plt.ylim((-0.5, cost.shape[1]-0.5))
            mfcc2_n=mfcc1
          #  mfcc2_n[0]=mfcc2[0]
            a=path[0]
            b=path[1]
            for l in range(1,len(path[0])):
                mfcc2_n[b[l]] = mfcc2[a[l]]
            with open(os.path.join(con_path,str(j)+'.pkl'), 'wb') as f:
                pickle.dump(mfcc2_n, f)
        print(i,j)

filepath = 'train/disentanglement/Aligned_audio_data/M030/'
for i in range(8):
    con_path='train/disentanglement/emotion_length/M030/'+str(i)
    if not os.path.exists(con_path):
        os.makedirs(con_path)

    for j in range(13):
        f=open(filepath+str(j)+'/'+str(i)+'.pkl','rb')
        mfcc=pickle.load(f)
        f.close()
        time_len = mfcc.shape[0]
        length = 0
        for input_idx in range(int((time_len-28)/4)+1):

            input_feat = mfcc[4*input_idx:4*input_idx+sample_len,:]

            with open(os.path.join(con_path,str(length)+'.pkl'), 'wb') as f:
                pickle.dump(input_feat, f)
            length+=1
            print(i,j,input_idx)



'''
#test
filepath='D:/codes/test_mfcc/'
for i in range(20):

    f=open('D:/codes/jixinya/SER/MFCC/M003_01_1_output_02/'+str(i)+'.pickle','rb')
    a=pickle.load(f)
#    plt.figure()
    f.close()
    ax = sns.heatmap(a, vmin=-100, vmax=100,cmap='rainbow')      #frames
  #  ax = sns.heatmap(a,center=0,cmap='rainbow')

    plt.savefig(filepath+str(i)+'.png')
    plt.close()
    print(i)



ax = sns.heatmap(a, vmin=-90, vmax=90,cmap='rainbow') #'rainbow''YlGnBu'
plt.savefig(filepath+str(i)+'.png')
plt.show()

fig = plt.figure() #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(a)  #绘制热力图，从-1到1 #, vmin=-90, vmax=90
fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
ticks = np.arange(0,9,1) #生成0-9，步长为1
ax.set_xticks(ticks)  #生成刻度
ax.set_yticks(ticks)
#ax.set_xticklabels(names) #生成x轴标签
#ax.set_yticklabels(names)
plt.show()


import matplotlib.pylab as plt
mfcc_features = a.T

plt.matshow(a)

plt.title('MFCC')

# get aligned MFCC
sample_len = 28
filepath='E:/MEAD/jixinya/Aligned_all_data/intensity1'
pathDir =  os.listdir(filepath)
for j in range(0,8):
    length=0
    outpath='E:/MEAD/jixinya/All_intensity/intensity1/'+str(j)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for i in range(0,13):
        path = os.path.join(filepath,str(i))

        mfcc_path = os.path.join(path,str(j)+'.pkl')
        f=open(mfcc_path,'rb')
        mfcc = pickle.load(f)
        f.close()
        time_len = mfcc.shape[0]
        for input_idx in range(int((time_len-28)/4)+1):
         #   target_idx = input_idx + sample_delay #14

            input_feat = mfcc[4*input_idx:4*input_idx+sample_len,:]

            with open(os.path.join(outpath,str(length)+'.pkl'), 'wb') as f:
                pickle.dump(input_feat, f)
            length+=1

    print('Emotion number %d, Total length %d ' %(j,length))

#check length
filepath='E:/MEAD/jixinya/Aligned_all_data/intensity3/'
pathDir =  os.listdir(filepath)
for j in range(0,8):
    time_len = 0
    for i in range(0,13):
        path = os.path.join(filepath,str(i))

        mfcc_path = os.path.join(path,str(j)+'.pkl')
        f=open(mfcc_path,'rb')
        mfcc = pickle.load(f)
        f.close()
        time_len += mfcc.shape[0]
        print(j,i,time_len)



ax = sns.heatmap(mfcc, vmin=-100, vmax=100,cmap='rainbow')
#check dtw results
#filepath='D:\codes\myaudio_aligned'
filepath='E:/MEAD/5.16_savee/Aligned/DC/'
path = 'D:/codes/test_mfcc/'
for i in range(1,3):
    f_path = os.path.join(filepath,str(i))
    for j in range(0,3):
        savepath = path+str(i)+'/'+str(j)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        mfcc_path = os.path.join(f_path,str(j)+'.pkl')
        f=open(mfcc_path,'rb')
        mfcc = pickle.load(f)
        f.close()
        for k in range(mfcc.shape[0]):
            x=mfcc[k]
            x = x.tolist()

            plt.ylim(-90, 90)
            plt.bar(range(len(x)), x,color='rgb')
            plt.savefig(os.path.join(savepath,str(k)+'.png'))
        #    plt.show()
            plt.close()
        print(i,j)

from moviepy.editor import *
path0=r'D:\codes\jixinya\DTW_check\0.mp4'
path1=r'D:\codes\jixinya\DTW_check\1.mp4'
path2=r'D:\codes\jixinya\DTW_check\2.mp4'
path3=r'D:\codes\jixinya\DTW_check\3.mp4'
path4=r'D:\codes\jixinya\DTW_check\4.mp4'
video0 = VideoFileClip(path0)
video1 = VideoFileClip(path1)
video2 = VideoFileClip(path2)
video3 = VideoFileClip(path3)
video4 = VideoFileClip(path4)
video = concatenate_videoclips([video0,video1,video2,video3,video4])
video.write_videofile(r'D:\codes\jixinya\DTW_check\all.mp4')

#my audio file test
filepath='D:/codes/myaudio/'
pathDir =  os.listdir(filepath)
con_path = 'D:/codes/myaudio_aligned/'
for j in range(len(pathDir)):
        audio_path = filepath+pathDir[j]
        if(j == 0):
            y1, sr1 = librosa.load(audio_path,sr=16000)
           # y1 = np.insert(y1, 0, np.zeros(1920))
           # y1 = np.append(y1, np.zeros(1920))
            mfcc = python_speech_features.mfcc(y1 , sr1 ,winstep=sample_interval)
            with open(con_path+str(j)+'.pkl', 'wb') as f:
                pickle.dump(mfcc, f)
        else:
            f=open(os.path.join(con_path,'0.pkl'),'rb')
            mfcc1 = pickle.load(f)
            y2, sr2 = librosa.load(audio_path,sr=16000)
           # y2 = np.insert(y2, 0, np.zeros(1920))
           # y2 = np.append(y2, np.zeros(1920))
            mfcc2 = python_speech_features.mfcc(y2 , sr2 ,winstep=sample_interval)
            dist, cost, acc_cost, path = dtw(mfcc2, mfcc1, dist=lambda x, y: norm(x - y, ord=1))

            plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
            plt.plot(path[0], path[1], 'w')
            plt.xlim((-0.5, cost.shape[0]-0.5))
            plt.ylim((-0.5, cost.shape[1]-0.5))
            mfcc2_n=mfcc1
          #  mfcc2_n[0]=mfcc2[0]
            a=path[0]
            b=path[1]
            for k in range(1,len(path[0])):
                mfcc2_n[b[k]] = mfcc2[a[k]]
            with open(con_path+str(j)+'.pkl', 'wb') as f:
                pickle.dump(mfcc2_n, f)
        print(j)

filepath='D:\codes\myaudio_aligned'
path = 'D:/codes/myaudio_image/'
for i in range(0,3):


    savepath = path+str(i)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    mfcc_path = os.path.join(filepath,str(i)+'.pkl')
    f=open(mfcc_path,'rb')
    mfcc = pickle.load(f)
    f.close()
    for k in range(mfcc.shape[0]):
        x=mfcc[k]
        x = x.tolist()

        plt.ylim(-90, 90)
        plt.bar(range(len(x)), x,color='rgb')
        plt.savefig(os.path.join(savepath,str(k)+'.png'))
        #    plt.show()
        plt.close()
    print(i)

#check number
import cv2
fps = 30           #保存视频的帧率
size = (432,288) #保存视频分辨率的大小

filepath = 'F:/computer/notebook/D/codes/test_mfcc/1/0/'
pathDir = os.listdir(filepath)
number = 0
for n in pathDir:
    number +=1
    video_path = 'E:/MEAD/M003_30fps/'+n+'/'
    if not os.path.exists(video_path):
            os.makedirs(video_path)
    video_path = video_path +'video.avi'
    video =cv2.VideoWriter(video_path,
            cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)

    path = 'E:/MEAD/M003/'+n+'/'
    filelist = os.listdir(path)
    for item in filelist:
        if item.endswith('.png'):
    #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
            item = path + item
            img = cv2.imread(item)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print(n,number)

path='/media/thea/其他/pca/overlap/norm/'
for i in range(len(apoints)):
    b=points[i].reshape(170,2)
    lab=os.path.join(path,str(i)+'.jpg')
    utils.plot_flmarks(b, lab, xLim, yLim, xLab, yLab, figsize=(10, 10))
    print('Generate %d jpg' % i)

#check overlap
def default_parameter_reader(flist):
    parameter_list = []
    name_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            name = line.split(' ')[-1]
            parameters = line.split(' ')[:-1]
            for i in range(len(parameters)):
                parameters[i] = float(parameters[i])
            parameter_list.append(parameters)
            name_list.append(name)
    return name_list,parameter_list


def parameter_reader(flist):
    parameter_list = []
 #   name_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
          #  name = line.split(' ')[-1]
            parameters = line.split(' ')
            for i in range(len(parameters)):
                parameters[i] = float(parameters[i])
            parameter_list.append(parameters)
        #    name_list.append(name)
    return parameter_list

font = {'size'   : 18}
def plot_flmarks(pts, lab, xLim, yLim, xLab, yLab, figsize=(10, 10)):
    if len(pts.shape) != 2:
        pts = np.reshape(pts, (pts.shape[0]/2, 2))

#    if pts.shape[0] == 20:
#        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
#        print (lookup)
#    else:
#        lookup = faceLmarkLookup

#    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(lookup))]

    plt.figure(figsize=figsize)
 #   plt.plot(pts[:,0], pts[:,1], 'ko', ms=4)
 #   cnt = 0
#    for refpts in lookup:
 #       lines[cnt].set_data([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]])
 #       cnt+=1
#        plt.plot([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]], 'k', ms=4)

    plt.plot(pts[:,0], pts[:,1], 'ko', ms=4)

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


flist = 'E:/MEAD/M003_01_1_output_03/crop_256/crop_lmk.txt'
name,parameters = default_parameter_reader(flist)
for i in range(len(name)):
    n=name[i]
    a= n.split('.')
    a[1] = a[1][-6:]
    n = a[0]+'_'+a[1]+'.'+a[2]
    name[i] = n[:-1]



filepath='E:/MEAD/M003_01_1_output_03/crop_256/frames/'
r_path='E:/MEAD/M003_01_1_output_03/crop_256/overlap/'
if not os.path.exists(r_path):
    os.makedirs(r_path)
pathDir = os.listdir(filepath)
count=0
for i in range(len(name)):
    path=os.path.join(filepath,name[i])
    p = parameters[i][6:]
    p = np.array(p)
    p = p.reshape(106,2)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    for k in range(len(p)):

        tup=(int(p[k,0]),int(p[k,1]))
        cv2.circle(img, tup, 0, (0,255,0 ), 2)
 #
    cv2.imwrite(r_path+'/'+str(i)+'.jpg',img)
  #  cv2.imshow('image',img)
  #  cv2.waitKey (10000) # 显示 10000 ms 即 10s 后消失
  #  cv2.destroyAllWindows()
    print(i)

path='E:/MEAD/M003_01_1_output_03/crop_256/landmark/'
if not os.path.exists(path):
    os.makedirs(path)
for i in range(len(name)):
    p = parameters[i][6:]
    p = np.array(p)
    p = p.reshape(106,2)
    lab=os.path.join(path,str(i)+'.jpg')
    plot_flmarks(p, lab, (0,255.0),(0,255.0),'x','y')
    print('Generate %d jpg' % i)

import matplotlib.pyplot as plt

flist = 'E:/MEAD/M003_01_1_output_03/crop_256/landmark.txt'
para = parameter_reader(flist)


filepath='E:/MEAD/M003_01_1_output_03/crop_256/frames/'
r_path='E:/MEAD/M003_01_1_output_03/crop_256/68_overlap/'
if not os.path.exists(r_path):
    os.makedirs(r_path)
pathDir = os.listdir(filepath)
count=0
for i in range(len(name)):
    path=os.path.join(filepath,name[i])
    p = para[i]
    p = np.array(p)*255
    p = p.reshape(68,2)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    for k in range(len(p)):

        tup=(int(p[k,0]),int(p[k,1]))
        cv2.circle(img, tup, 0, (0,255,0 ), 2)
 #
    cv2.imwrite(r_path+'/'+str(i)+'.jpg',img)
  #  cv2.imshow('image',img)
  #  cv2.waitKey (10000) # 显示 10000 ms 即 10s 后消失
  #  cv2.destroyAllWindows()
    print(i)

path='E:/MEAD/M003_01_1_output_03/crop_256/68_landmark/'
if not os.path.exists(path):
    os.makedirs(path)
for i in range(len(name)):
    p = para[i]
    p = np.array(p)*255
    p = p.reshape(68,2)
    lab=os.path.join(path,str(i)+'.jpg')
    plot_flmarks(p, lab, (0,255.0),(0,255.0),'x','y')
    print('Generate %d jpg' % i)

#savee
filepath = 'E:/MEAD/5.16_savee/AudioData'
emotion = ['a', 'd', 'f', 'h', 'n', 'sa' ,'su' ]
pathDir = os.listdir(filepath)
name = 'KL'
filepath = os.path.join(filepath,name)
for i in range(1,16):


    con_path='E:/MEAD/5.16_savee/Aligned/'+name+'/'+str(i)
    if not os.path.exists(con_path):
        os.makedirs(con_path)
    for j in range(len(emotion)):
        audio_path = os.path.join(filepath,emotion[j]+'{:02d}'.format(i)+'.wav')
        if(j == 0):
            y1, sr1 = librosa.load(audio_path,sr=16000)
            y1 = np.insert(y1, 0, np.zeros(1920))
            y1 = np.append(y1, np.zeros(1920))
            mfcc = python_speech_features.mfcc(y1 , sr1 ,winstep=sample_interval)
            with open(con_path+'/'+str(j)+'.pkl', 'wb') as f:
                pickle.dump(mfcc, f)
        else:
            f=open(os.path.join(con_path,'0.pkl'),'rb')
            mfcc1 = pickle.load(f)
            y2, sr2 = librosa.load(audio_path,sr=16000)
            y2 = np.insert(y2, 0, np.zeros(1920))
            y2 = np.append(y2, np.zeros(1920))
            mfcc2 = python_speech_features.mfcc(y2 , sr2 ,winstep=sample_interval)
            dist, cost, acc_cost, path = dtw(mfcc2, mfcc1, dist=lambda x, y: norm(x - y, ord=1))

            plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
            plt.plot(path[0], path[1], 'w')
            plt.xlim((-0.5, cost.shape[0]-0.5))
            plt.ylim((-0.5, cost.shape[1]-0.5))
            mfcc2_n=mfcc1
          #  mfcc2_n[0]=mfcc2[0]
            a=path[0]
            b=path[1]
            for k in range(1,len(path[0])):
                mfcc2_n[b[k]] = mfcc2[a[k]]
            with open(os.path.join(con_path,str(j)+'.pkl'), 'wb') as f:
                pickle.dump(mfcc2_n, f)
        print(i,j)


# get aligned MFCC
sample_len = 28
filepath='E:/MEAD/5.16_savee/Aligned_same_length/'
pathDir = os.listdir(filepath)
name = 'KL'
filepath = os.path.join(filepath,name)
for j in range(0,7):
    length=0
    outpath='E:/MEAD/5.16_savee/emotion_same_length/'+name+'/'+str(j)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for i in range(1,16):
        path = os.path.join(filepath,str(i))

        mfcc_path = os.path.join(path,str(j)+'.pkl')
        f=open(mfcc_path,'rb')
        mfcc = pickle.load(f)
        f.close()
        time_len = mfcc.shape[0]
        for input_idx in range(int((time_len-28)/4)+1):
         #   target_idx = input_idx + sample_delay #14

            input_feat = mfcc[4*input_idx:4*input_idx+sample_len,:]

            with open(os.path.join(outpath,str(length)+'.pkl'), 'wb') as f:
                pickle.dump(input_feat, f)
            length+=1

    print('Emotion number %d, Total length %d ' %(j,length))

#savee_same_length
filepath = 'E:/MEAD/5.16_savee/AudioData'
emotion = ['a', 'd', 'f', 'h', 'n', 'sa' ,'su' ]
pathDir = os.listdir(filepath)
name = 'JE'
filepath = os.path.join(filepath,name)
for i in range(1,16):


    con_path='E:/MEAD/5.16_savee/Aligned_same_length/'+'DC'+'/'+str(i)
    save_path = 'E:/MEAD/5.16_savee/Aligned_same_length/'+name+'/'+str(i)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for j in range(len(emotion)):
        audio_path = os.path.join(filepath,emotion[j]+'{:02d}'.format(i)+'.wav')


        f=open(os.path.join(con_path,'0.pkl'),'rb')
        mfcc1 = pickle.load(f)
        y2, sr2 = librosa.load(audio_path,sr=16000)
        y2 = np.insert(y2, 0, np.zeros(1920))
        y2 = np.append(y2, np.zeros(1920))
        mfcc2 = python_speech_features.mfcc(y2 , sr2 ,winstep=sample_interval)
        dist, cost, acc_cost, path = dtw(mfcc2, mfcc1, dist=lambda x, y: norm(x - y, ord=1))

        plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.xlim((-0.5, cost.shape[0]-0.5))
        plt.ylim((-0.5, cost.shape[1]-0.5))
        mfcc2_n=mfcc1
          #  mfcc2_n[0]=mfcc2[0]
        a=path[0]
        b=path[1]
        for k in range(1,len(path[0])):
            mfcc2_n[b[k]] = mfcc2[a[k]]
        with open(os.path.join(save_path,str(j)+'.pkl'), 'wb') as f:
            pickle.dump(mfcc2_n, f)
        print(i,j)
'''




