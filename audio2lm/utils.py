import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.animation as manimation
import matplotlib.lines as mlines
from mpl_toolkits import mplot3d
import argparse, os, fnmatch, shutil
from collections import OrderedDict
from scipy.spatial import procrustes
from torch.autograd import Variable
import torch
import numpy as np
import cv2
import math
import copy
import librosa
import dlib
import subprocess
# from keras import backend as K
from tqdm import tqdm
from skimage import transform as tf
import torchvision.transforms as transforms
import random
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def melSpectra(y, sr, wsize, hsize):
    cnst = 1+(int(sr*wsize)/2)
    y_stft_abs = np.abs(librosa.stft(y,
                                  win_length = int(sr*wsize),
                                  hop_length = int(sr*hsize),
                                  n_fft=int(sr*wsize)))/cnst

    melspec = np.log(1e-16 + librosa.feature.melspectrogram(sr=sr, 
                                             S=y_stft_abs**2,
                                             n_mels=64))
    return melspec


def draw_mouth(landmark, width, height):
    landmark = landmark.reshape(106*2,)
    # draw mouth from mouth landmarks, landmarks: mouth landmark points, format: x1, y1, x2, y2, ..., x20,
    heatmap = 255*np.ones((width, height, 3), dtype=np.uint8)
    circle_color = (255, 0, 0)
    line_color = (0, 255, 0)
   
    def draw_line(start_idx, end_idx):
        for pts_idx in range(start_idx, end_idx):
            cv2.line(heatmap, (int(landmark[pts_idx * 2]), int(landmark[pts_idx * 2 + 1])),
                     (int(landmark[pts_idx * 2 + 2]), int(landmark[pts_idx * 2 + 3])), line_color, 3)
    draw_line(0, 32)     # face 
    # EYEBROW + MOUTH
    draw_line(33, 37)     
    draw_line(38, 42)     
    draw_line(64, 67)     
    draw_line(68, 71) 
    
    draw_line(84, 90)     # upper outer
    draw_line(96, 100)   # upper inner
    draw_line(100, 103)   # lower inner
    draw_line(90, 95)    # lower outer
        
    cv2.line(heatmap, (int(landmark[33 * 2]), int(landmark[33 * 2 + 1])),
             (int(landmark[64 * 2]), int(landmark[64 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[37 * 2]), int(landmark[37* 2 + 1])),
             (int(landmark[67 * 2]), int(landmark[67 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[38 * 2]), int(landmark[38 * 2 + 1])),
             (int(landmark[68 * 2]), int(landmark[68 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[42 * 2]), int(landmark[42* 2 + 1])),
             (int(landmark[71 * 2]), int(landmark[71 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[96 * 2]), int(landmark[96 * 2 + 1])),
             (int(landmark[103 * 2]), int(landmark[103 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[84 * 2]), int(landmark[84 * 2 + 1])),
             (int(landmark[95 * 2]), int(landmark[95 * 2 + 1])), thickness=3, color=line_color)
    #LEFT EYE
    draw_line(52, 53)   # lower inner
    draw_line(54, 56)    # lower outer
    
    cv2.line(heatmap, (int(landmark[53 * 2]), int(landmark[53 * 2 + 1])),
             (int(landmark[72 * 2]), int(landmark[72 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[72 * 2]), int(landmark[72* 2 + 1])),
             (int(landmark[54 * 2]), int(landmark[54 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[73 * 2]), int(landmark[73 * 2 + 1])),
             (int(landmark[56 * 2]), int(landmark[56 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[57 * 2]), int(landmark[57* 2 + 1])),
             (int(landmark[73 * 2]), int(landmark[73 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[52 * 2]), int(landmark[52 * 2 + 1])),
             (int(landmark[57 * 2]), int(landmark[57 * 2 + 1])), thickness=3, color=line_color)
    #RIGHT EYE
    draw_line(58, 59)   # lower inner
    draw_line(60, 62)    # lower outer
    
    cv2.line(heatmap, (int(landmark[59 * 2]), int(landmark[59 * 2 + 1])),
             (int(landmark[75 * 2]), int(landmark[75 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[75 * 2]), int(landmark[75* 2 + 1])),
             (int(landmark[60 * 2]), int(landmark[60 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[76 * 2]), int(landmark[76 * 2 + 1])),
             (int(landmark[62 * 2]), int(landmark[62 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[63 * 2]), int(landmark[63* 2 + 1])),
             (int(landmark[76 * 2]), int(landmark[76 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[58 * 2]), int(landmark[58 * 2 + 1])),
             (int(landmark[63 * 2]), int(landmark[63 * 2 + 1])), thickness=3, color=line_color)
    
    #NOSE
    draw_line(43, 46)   # lower inner
    draw_line(47, 51)    # lower outer
    
    cv2.line(heatmap, (int(landmark[78 * 2]), int(landmark[78 * 2 + 1])),
             (int(landmark[80 * 2]), int(landmark[80 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[80 * 2]), int(landmark[80* 2 + 1])),
             (int(landmark[82 * 2]), int(landmark[82 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[47 * 2]), int(landmark[47 * 2 + 1])),
             (int(landmark[82 * 2]), int(landmark[82 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[79 * 2]), int(landmark[79* 2 + 1])),
             (int(landmark[81 * 2]), int(landmark[81 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[81 * 2]), int(landmark[81 * 2 + 1])),
             (int(landmark[83 * 2]), int(landmark[83 * 2 + 1])), thickness=3, color=line_color)
    cv2.line(heatmap, (int(landmark[51 * 2]), int(landmark[51 * 2 + 1])),
             (int(landmark[83 * 2]), int(landmark[83 * 2 + 1])), thickness=3, color=line_color)
    # draw keypoints
    for pts_idx in range(106):
        cv2.circle(heatmap, (int(landmark[pts_idx * 2]), int(landmark[pts_idx * 2 + 1])), radius=3, thickness=-1,
                   color=circle_color)
    return heatmap

class VideoWriter(object):
    def __init__(self, path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.path = path
        self.out = cv2.VideoWriter(self.path, fourcc, fps, (width, height))

    def write_frame(self, frame):
        self.out.write(frame)
        
    def end(self):
        self.out.release()

        

def crop_image(image_path, detector, shape, predictor):
    

  
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
      
        (x, y, w, h) = rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)

        r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        
        roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)
        scale =  163. / (2 * r)
       
        shape = ((shape - np.array([new_x,new_y])) * scale)
    
        return roi, shape 

      
def image_to_video(sample_dir = None, video_name = None):
    
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.png -c:v libx264 -y -vf scale=640:640 ' + video_name 
    #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print (command)
    os.system(command)

def add_audio(video_name=None, audio_dir = None):

    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.avi','.mov')
    #ffmpeg -i /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/new/audio/obama.wav -codec copy -c:v libx264 -c:a aac -b:a 192k  -shortest -y /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mov
    # ffmpeg -i gan_r_high_fake.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/audio/obama.wav -vcodec copy  -acodec copy -y   gan_r_high_fake.mov

    print (command)
    os.system(command)

def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx





def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    path_to_file = os.path.join('../results', file_name + '.jpg')
    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    cv2.imwrite(path_to_file, gradient)


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name+'_Cam_Grayscale.jpg')
    print (path_to_file)
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (128, 128))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name+'_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (128, 128))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    im_as_ten = torch.FloatTensor(im_as_ten)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.5, -0.5, -0.5]
    reverse_std = [1/0.5, 1/0.5, 1/0.5]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

def check_volume(speech,sr):
    length = int(len(speech)/640)
    speech = speech[:length*640]
    speech = speech.reshape(-1,640)
    clip = np.abs(speech.sum(1))
    clip = clip > 0.1
    i = 0
    while i < len(clip):
        if(clip[i] == 0):
            n = 0
            while((i+n) < len(clip) and clip[i+n] == 0):
                n = n+1
            if (n < 20):
                clip[i:i+n] = 1
            i = i+n
        else:
            if (i+1) < len(clip) and clip[i+1] ==0 and i > 0 and clip[i-1] ==0 :
                clip[i] = 0
            i = i+1
    return clip

def change_mouth(fake_lmark, clip):
    if len(fake_lmark) < len(clip):
        clip = clip[:len(fake_lmark)]
    index = 0
    s = 1
    for i in range(len(fake_lmark)):
        lmark = fake_lmark[i]
        if (lmark[102][1] - lmark[98][1]) < s:
            s = lmark[102][1] - lmark[98][1]
            index = i
    close_mouth = fake_lmark[index]
    c = np.array(clip, dtype = float)
    for i in range(1, len(c)):
        if c[i] == 0:
            if c[i-1] == 1:
                fake_lmark[i] = 0.8*fake_lmark[i-1] + 0.2*close_mouth
                c[i] = 0.8
                fake_lmark[i+1] = 0.6*fake_lmark[i-1] + 0.4*close_mouth
                c[i+1] = 0.6
                fake_lmark[i+2] = 0.4*fake_lmark[i-1] + 0.6*close_mouth
                c[i+2] = 0.4
                fake_lmark[i+3] = 0.2*fake_lmark[i-1] + 0.8*close_mouth
                c[i+3] = 0.2
            elif ((i+1)< len(c)) and (c[i+1] == 1):
                fake_lmark[i] = 0.8*fake_lmark[i+1] + 0.2*close_mouth
                c[i] = 0.8
                fake_lmark[i-1] = 0.6*fake_lmark[i+1] + 0.4*close_mouth
                c[i-1] = 0.6
                fake_lmark[i-2] = 0.4*fake_lmark[i+1] + 0.6*close_mouth
                c[i-2] = 0.4
                fake_lmark[i-3] = 0.2*fake_lmark[i+1] + 0.8*close_mouth
                c[i-3] = 0.2
    for i in range(len(c)):
        if c[i] == 0:
            ratio = random.uniform(0.9,1)
            fake_lmark[i] = (1-ratio)*fake_lmark[i] + ratio*close_mouth
    return fake_lmark
    
    
    
        
        
        

def main():
    return

if __name__ == "__main__":
    main()