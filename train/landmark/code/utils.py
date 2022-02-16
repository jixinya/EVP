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

font = {'size'   : 18}
# mpl.rc('font', **font)

# Lookup tables for drawing lines between points
#Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
#         [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
#         [66, 67], [67, 60]]

Nose = [[43, 78], [43, 79], [43, 44], [44, 45], [45, 46], [46, 49], [45, 80], \
        [80, 82], [45, 81], [81, 83], [47, 82],[47, 48], [48, 49], [49, 50], [50, 51],\
        [51, 83]]

leftBrow = [[33, 34], [34, 35], [35, 36], [36, 37],[33,64], [64, 65], \
            [65, 66], [66, 67],[67, 37]]
rightBrow = [[38, 39], [39, 40], [40, 41], [41, 42],[38,68], [68, 69], \
            [69, 70], [70, 71],[71, 42]]

leftEye = [[52, 53], [53,72],[54, 72], [54, 55], [55, 56], [56, 73],[57,73], [52, 57]]
rightEye = [[58, 59], [59, 75],[60,75], [60, 61], [61, 62], [62, 76],[76,63], [58, 63]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16],[16, 17], \
         [17, 18], [18, 19], [19, 20],[20, 21], [21, 22],\
         [22, 23], [23, 24],[24, 25], [25, 26], [26, 27], \
         [27, 28],[28, 29], [29, 30], [30, 31], [31, 32]]

faceLmarkLookup =  Nose + leftBrow + rightBrow + leftEye + rightEye + other

#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor('../basics/shape_predictor_68_face_landmarks.dat')
#ms_img = np.load('../basics/mean_shape_img.npy')
#ms_norm = np.load('../basics/mean_shape_norm.npy')
#S = np.load('../basics/S.npy')

#MSK = np.reshape(ms_norm, [1, 68*2])
#SK = np.reshape(S, [1, S.shape[0], 68*2])
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

def normLmarks(lmarks):
    norm_list = []
    idx = -1
    max_openness = 0.2
    mouthParams = np.zeros((1, 100))
    mouthParams[:, 1] = -0.06
    tmp = copy.deepcopy(MSK)
    tmp[:, 48*2:] += np.dot(mouthParams, SK)[0, :, 48*2:]
    open_mouth_params = np.reshape(np.dot(S, tmp[0, :] - MSK[0, :]), (1, 100))
    print (lmarks.shape)
    for i in range(lmarks.shape[0]):
        mtx1, mtx2, disparity = procrustes(ms_img, lmarks[i, :, :])
        mtx1 = np.reshape(mtx1, [1, 136])
        mtx2 = np.reshape(mtx2, [1, 136])
        norm_list.append(mtx2[0, :])
    pred_seq = []
    init_params = np.reshape(np.dot(S, norm_list[idx] - mtx1[0, :]), (1, 100))
    for i in range(lmarks.shape[0]):
        params = np.reshape(np.dot(S, norm_list[i] - mtx1[0, :]), (1, 100)) - init_params - open_mouth_params
        predicted = np.dot(params, SK)[0, :, :] + MSK
        pred_seq.append(predicted[0, :])
    return np.array(pred_seq), np.array(norm_list), 1

def write_video_wpts_wsound_unnorm(frames, sound, fs, path, fname, xLim, yLim):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print ('Exp')

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=25, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(dt))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in faceLmarkLookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_ws.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))

def write_video_wpts_wsound(frames, sound , fs, path, fname, xLim, yLim):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print ('Exp')

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=25, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    rect = (0, 0, 600, 600)
    
    if frames.shape[1] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print (lookup)
    else:
        lookup = faceLmarkLookup

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(lookup))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in lookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -y -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_ws.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

  #  os.remove(os.path.join(path, fname+'.mp4'))
   # os.remove(os.path.join(path, fname+'.wav'))


def plot_flmarks(pts, lab, xLim, yLim, xLab, yLab, figsize=(10, 10)):
    if len(pts.shape) != 2:
        pts = np.reshape(pts, (pts.shape[0]/2, 2))

  #  if pts.shape[0] == 20:
  #      lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
   #     print (lookup)
   # else:
   #     lookup = faceLmarkLookup

    plt.figure(figsize=figsize)
    plt.plot(pts[:,0], pts[:,1], 'ko', ms=4)
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

# def main():
#     return

# if __name__ == "__main__":
#     main()


def create_example(config, img_path = None):
    try:
        detector = dlib.get_frontal_face_detector()
        shape = os.path.join(config.data_path,'shape_predictor_68_face_landmarks.dat')
        predictor = dlib.shape_predictor(shape)
        roi,landmark = crop_image(img_path, detector, shape, predictor)
        if np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9 :
            template = np.load(os.path.join(config.data_path,'base_68.npy'))
        else:
            template = np.load(os.path.join(config.data_path,'base_68_close.npy'))
        pts2 = np.float32(template[27:45,:])
        pts1 = np.float32(landmark[27:45,:])
        tform = tf.SimilarityTransform()
        tform.estimate(pts2,pts1)
        dst = tf.warp(roi,tform,output_shape = (163,163))
        dst = np.array(dst * 255 , dtype = np.uint8)
        dst = dst[1:129,1:129,:]
        cv2.imwrite(img_path.replace('.jpg','_region.jpg'), dst)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            
            # for j in range(len(shape)):
            #     x=int(shape[j][1])
            #     y =int(shape[j][0])
            #     cv2.circle(dst, (y, x), 1, (0, 0, 255), -1)
            # cv2.imwrite(img_path.replace('.jpg','_lm.jpg'),dst)
            np.save(img_path.replace('.jpg','.npy'),shape)
        shape, _, _ = normLmarks(shape.reshape(1,68,2))
        example_img = cv2.imread(img_path.replace('.jpg','_region.jpg'))
        
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])        
        example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
        example_img = transform(example_img)
        example_img = torch.FloatTensor(example_img)

        return example_img, shape
    except:
        image = cv2.imread(img_path)
        example_img = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])        
        example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
        example_img = transform(example_img)
        example_img = torch.FloatTensor(example_img)
        example_landmark = np.load('../image/musk1.npy').reshape(1, 68, 2)

        example_landmark, _, _  = normLmarks(example_landmark)


        return example_img, example_landmark



        
        

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

    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
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



def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result


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


# def plot_face( lm):


#     inds_mouth = [60, 61, 62, 63, 64, 65, 66, 67, 60]
#     inds_top_teeth = [48, 54, 53, 52, 51, 50, 49, 48]
#     inds_bottom_teeth = [4, 12, 10, 6, 4]
#     inds_skin = [0, 1, 2, 3, 4, 5, 6, 7, 8,
#                     57, 58, 59, 48, 49, 50, 51, 52, 52, 53, 54, 55, 56, 57,
#                     8, 9, 10, 11, 12, 13, 14, 15, 16,
#                     45, 46, 47, 42, 43, 44, 45,
#                     16, 71, 70, 69, 68, 0,
#                     36, 37, 38, 39, 40, 41, 36, 0]
#     inds_lips = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48,
#                     60, 67, 66, 65, 64, 63, 62, 61, 60, 48]
#     inds_nose = [[27, 28, 29, 30, 31, 27],
#                     [30, 31, 32, 33, 34, 35, 30],
#                     [27, 28, 29, 30, 35, 27]]
#     inds_brows = [[17, 18, 19, 20, 21],
#                     [22, 23, 24, 25, 26]]
#     plt.axes().set_aspect('equal', 'datalim')
#     # make some eyes
#     theta = np.linspace(0, 2 * np.pi, 100)
#     circle = np.transpose([np.cos(theta), np.sin(theta)])
#     for inds_eye in [[37, 38, 40, 41], [43, 44, 46, 47]]:
#         plt.fill(.013 * circle[:, 0] + lm[inds_eye, 0].mean(),
#                 .013 * circle[:, 1] - lm[inds_eye, 1].mean(),
#                 color=[0, 0.5, 0], lw=0)
#         plt.fill(.005 * circle[:, 0] + lm[inds_eye, 0].mean(),
#                 .005 * circle[:, 1] - lm[inds_eye, 1].mean(),
#                 color=[0, 0, 0], lw=0)
#     plt.plot(.01 * circle[:, 0], .01 * circle[:, 1], color=[0, 0.5, 0], lw=0)
#     # make the teeth
#     # nose bottom to top teeth: 0.037
#     # chin bottom to bottom teeth: .088
#     plt.fill(lm[inds_mouth, 0], -lm[inds_mouth, 1], color=[0, 0, 0], lw=0)
#     # plt.fill(lm[inds_top_teeth, 0], -lm[inds_top_teeth, 1], color=[1, 1, 0.95], lw=0)
#     # plt.fill(lm[inds_bottom_teeth, 0], -lm[inds_bottom_teeth, 1], color=[1, 1, 0.95], lw=0)

#     # make the rest
#     skin_color = np.array([0.7, 0.5, 0.3])
#     plt.fill(lm[inds_skin, 0], -lm[inds_skin, 1], color=skin_color, lw=0)
#     for ii, color_shift in zip(inds_nose, [-0.05, -0.1, 0.06]):
#         plt.fill(lm[ii, 0], -lm[ii, 1], color=skin_color + color_shift, lw=0)
#     plt.fill(lm[inds_lips, 0], -lm[inds_lips, 1], color=[0.7, 0.3, 0.2], lw=0)

#     for ib in inds_brows:
#         plt.plot(lm[ib, 0], -lm[ib, 1], color=[0.3, 0.2, 0.05], lw=4)

#     plt.xlim(-0.15, 0.15)
#     plt.ylim(-0.2, 0.18)




class facePainter():
    inds_mouth = [60, 61, 62, 63, 64, 65, 66, 67, 60]
    inds_top_teeth = [48, 54, 53, 52, 51, 50, 49, 48]
    inds_bottom_teeth = [4, 12, 10, 6, 4]
    inds_skin = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                    57, 58, 59, 48, 49, 50, 51, 52, 52, 53, 54, 55, 56, 57,
                    8, 9, 10, 11, 12, 13, 14, 15, 16,
                    45, 46, 47, 42, 43, 44, 45,
                    16, 71, 70, 69, 68, 0,
                    36, 37, 38, 39, 40, 41, 36, 0]
    inds_lips = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48,
                    60, 67, 66, 65, 64, 63, 62, 61, 60, 48]
    inds_nose = [[27, 28, 29, 30, 31, 27],
                    [30, 31, 32, 33, 34, 35, 30],
                    [27, 28, 29, 30, 35, 27]]
    inds_brows = [[17, 18, 19, 20, 21],
                    [22, 23, 24, 25, 26]]

    def __init__(self, lmarks, speech ):
        lmarks = np.concatenate((lmarks,
                                lmarks[:, [17, 19, 24, 26], :]), 1)[..., :2]
        lmarks[:, -4:, 1] += -0.03
        # lm = lmarks.mean(0)
        # lm = lmarks[600]

        self.lmarks = lmarks
        self.speech = speech

    def plot_face(self, lm):
        plt.axes().set_aspect('equal', 'datalim')

        # make some eyes
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.transpose([np.cos(theta), np.sin(theta)])
        for self.inds_eye in [[37, 38, 40, 41], [43, 44, 46, 47]]:
            plt.fill(.013 * circle[:, 0] + lm[self.inds_eye, 0].mean(),
                    .013 * circle[:, 1] - lm[self.inds_eye, 1].mean(),
                    color=[0, 0.5, 0], lw=0)
            plt.fill(.005 * circle[:, 0] + lm[self.inds_eye, 0].mean(),
                    .005 * circle[:, 1] - lm[self.inds_eye, 1].mean(),
                    color=[0, 0, 0], lw=0)
        plt.plot(.01 * circle[:, 0], .01 * circle[:, 1], color=[0, 0.5, 0], lw=0)
        # make the teeth
        # nose bottom to top teeth: 0.037
        # chin bottom to bottom teeth: .088
        plt.fill(lm[self.inds_mouth, 0], -lm[self.inds_mouth, 1], color=[0, 0, 0], lw=0)
        # plt.fill(lm[inds_top_teeth, 0], -lm[inds_top_teeth, 1], color=[1, 1, 0.95], lw=0)
        # plt.fill(lm[inds_bottom_teeth, 0], -lm[inds_bottom_teeth, 1], color=[1, 1, 0.95], lw=0)

        # make the rest
        skin_color = np.array([0.7, 0.5, 0.3])
        plt.fill(lm[self.inds_skin, 0], -lm[self.inds_skin, 1], color=skin_color, lw=0)
        for ii, color_shift in zip(self.inds_nose, [-0.05, -0.1, 0.06]):
            plt.fill(lm[ii, 0], -lm[ii, 1], color=skin_color + color_shift, lw=0)
        plt.fill(lm[self.inds_lips, 0], -lm[self.inds_lips, 1], color=[0.7, 0.3, 0.2], lw=0)

        for ib in self.inds_brows:
            plt.plot(lm[ib, 0], -lm[ib, 1], color=[0.3, 0.2, 0.05], lw=4)

        plt.xlim(-0.15, 0.15)
        plt.ylim(-0.2, 0.18)

    def write_video(self, frames, sound, fs, path, fname, xLim, yLim):
        try:
            os.remove(os.path.join(path, fname+'.mp4'))
            os.remove(os.path.join(path, fname+'.wav'))
            os.remove(os.path.join(path, fname+'_ws.mp4'))
        except:
            print ('Exp')

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=25, metadata=metadata)

        fig = plt.figure(figsize=(10, 10))
        # l, = plt.plot([], [], 'ko', ms=4)

        # plt.xlim(xLim)
        # plt.ylim(yLim)

        librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

        with writer.saving(fig, os.path.join(path, fname+'.mp4'), 100):
            # plt.gca().invert_yaxis()
            for i in tqdm(range(frames.shape[0])):
                self.plot_face(frames[i, :, :])
                writer.grab_frame()
                plt.clf()
                # plt.close()

        cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_.mp4'
        subprocess.call(cmd, shell=True) 
        print('Muxing Done')

        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))

    def paintFace(self, path, fname):
        self.write_video(self.lmarks, self.speech, 8000, path, fname, [-0.15, 0.15], [-0.2, 0.18])

def main():
    return

if __name__ == "__main__":
    main()