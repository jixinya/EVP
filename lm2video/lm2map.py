#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 16:51:39 2020

@author: asus
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:46:43 2020

@author: asus
"""
import cv2
import torch
import torch.nn as nn
import pickle
import numpy as np
#from tqdm import tqdm
import os

import torch
import matplotlib.pyplot as plt
from filter1 import OneEuroFilter
import shutil

class Exp2Ldmk(nn.Module):
    def __init__(self, params):
        super(Exp2Ldmk, self).__init__()
        self.batch_size = 1
        # landmark pca
        # self.pca = pickle.load(open(os.path.join(params['data_root'], 'stat.pickle'), 'rb'))['pca']
        self.load_param(params)

    def load_param(self, params):
        # load weight_shape and weight_exp
        with open('./base_weight.pickle', 'rb') as f:
            data = pickle.load(f)
        self.base = np.array(data['base'], dtype=np.float32)
        self.weight = np.array(data['weight'], dtype=np.float32)
        # load 24 curve idx
        # self.curve_idx = []
        # with open(paths['curve'], 'r') as f:
        #     lines = f.readlines()
        # for line in lines:
        #     info = line.strip().split()
        #     self.curve_idx.append([int(it) for it in info])
        # # load 33 contour curve idx
        # self.contour_idx = []
        # with open(paths['contour'], 'r') as f:
        #     lines = f.readlines()
        # for line in lines:
        #     info = line.strip().split()
        #     self.contour_idx.append([int(it) for it in info])
        # # load default ldmk idx (frontal face)
        # with open(paths['ldmk_idx'], 'r') as f:
        #     line = f.readline()
        # info = line.strip().split()
        self.ldmk_idx = [int(it) for it in info]
        # allocate variable space
        self.base_ = torch.tensor(self.base, dtype=torch.float32).cuda()  # 3*53215
        self.weight_ = torch.tensor(self.weight, dtype=torch.float32).cuda()  # 3*53215*228
        self.vertex_ = torch.zeros(self.batch_size, 3, 53215, dtype=torch.float32, requires_grad=True).cuda()
        self.pt2d_ = torch.zeros(self.batch_size, 2, 53215, dtype=torch.float32, requires_grad=True).cuda()
        self.ldmk_ = torch.zeros(self.batch_size, 2 * len(self.ldmk_idx), dtype=torch.float32,
                                 requires_grad=True).cuda()
        # self.ldmk_pca_ = torch.zeros(self.batch_size, params['shape_dim'], dtype=torch.float32, requires_grad=True).cuda()
        self.rotation_mat_ = torch.zeros(self.batch_size, 3, 3, dtype=torch.float32, requires_grad=True).cuda()
        # self.ldmk_mean_ = torch.tensor(self.pca.mean_, dtype=torch.float32, requires_grad=True).cuda()
        # self.ldmk_components_ = torch.tensor(self.pca.components_, dtype=torch.float32, requires_grad=True).cuda()

    def init_tensor(self):
        # detach as tensors for forward
        self.base = self.base_.detach()
        self.weight = self.weight_.detach()
        self.vertex = self.vertex_.detach()
        self.pt2d = self.pt2d_.detach()
        self.ldmk = self.ldmk_.detach()
        # self.ldmk_pca = self.ldmk_pca_.detach()
        self.rotation_mat = self.rotation_mat_.detach()
        # self.ldmk_mean = self.ldmk_mean_.detach()
        # self.ldmk_components = self.ldmk_components_.detach()

    def forward(self, shape, exp, pose):
        # initialize tensors, allocate memeory
        self.init_tensor()
        # concat shape and exp coefficient into param
        self.param = torch.cat((shape, exp), dim=1)
        # get rotation matrix
        # self.rotation_mat = self.rotation(pose[:, :3]).detach().cuda()
        for i in range(self.batch_size):
            self.vertex[i] = self.base + torch.matmul(self.weight, self.param[i])

        return self.vertex[0]
        # for i in range(self.batch_size):
        #    self.ldmk_pca[i] = self.pca_transform(self.ldmk[i])
        # return self.ldmk_pca





def rotation(rotation_mat, quaternion):
    assert quaternion.shape[1] == 3
    assert quaternion.shape[0] == 1
    for i in range(1):
        angle = torch.sqrt(quaternion[i, 0] ** 2 + quaternion[i, 1] ** 2 + quaternion[i, 2] ** 2)
        c, s = torch.cos(angle), torch.sin(angle)
        x, y, z = quaternion[i, 0] / angle, quaternion[i, 1] / angle, quaternion[i, 2] / angle
        xy, yz, zx = x * y, y * z, z * x
        x2, y2, z2 = x ** 2, y ** 2, z ** 2
        rotation_mat[i, 0, 0], rotation_mat[i, 1, 0], rotation_mat[i, 2, 0] = \
            (1 - c) * x2 + c, (1 - c) * xy + s * z, (1 - c) * zx - s * y
        rotation_mat[i, 0, 1], rotation_mat[i, 1, 1], rotation_mat[i, 2, 1] = \
            (1 - c) * xy - s * z, (1 - c) * y2 + c, (1 - c) * yz + s * x
        rotation_mat[i, 0, 2], rotation_mat[i, 1, 2], rotation_mat[i, 2, 2] = \
            (1 - c) * zx + s * y, (1 - c) * yz - s * x, (1 - c) * z2 + c
        rotation_mat[i, :, 1:] = -1 * rotation_mat[i, :, 1:]
    return rotation_mat

def ldmk3d_2d(pose, vertex):
    rotation_mat_ = torch.zeros(1, 3, 3, dtype=torch.float32, requires_grad=True).cuda()
    # get rotation matrix
    rotation_mat = rotation(rotation_mat_, pose[:, :3]).detach().cuda()
    # project 3d point into 2d point

    rotate_vertex = torch.matmul(rotation_mat[0], vertex)
    pt2d_x = rotate_vertex[0] * pose[0, 3] + pose[0, 4]
    pt2d_y = rotate_vertex[1] * pose[0, 3] + pose[0, 5]
    return pt2d_x, pt2d_y
'''
def ldmk3d_front(pose, vertex):
 #   rotation_mat_ = torch.zeros(1, 3, 3, dtype=torch.float32, requires_grad=True).cuda()
    # get rotation matrix
 #   rotation_mat = rotation(rotation_mat_, pose[:, :3]).detach().cuda()
    # rotation_mat = torch.zeros(1, 3, 3, dty1pe=torch.float32).cuda()
    rotation_mat[0,0,0] = 1
    rotation_mat[0,1,1] = -1
    rotation_mat[0,2,2] = -1
    # project 3d point into 2d point

    rotate_vertex = torch.matmul(rotation_mat[0], vertex)
    pt2d_x = rotate_vertex[0] * pose[0, 3] + 128#pose[0, 4]
    pt2d_y = rotate_vertex[1] * pose[0, 3] + 128#pose[0, 5]
    return pt2d_x, pt2d_y
'''

def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T

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

def default_parameter_reader(flist):
    parameter_list = []
    name_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            name = line.split(' ')[-1]
            parameters = line.split(' ')[:-1]
            for i in range(len(parameters)):
                parameters[i] = float(parameters[i])
          #  b=np.array(parameters)
            
          #  x = b[:106].reshape(106,1)
          #  y = b[106:].reshape(106,1)
          #  c = np.concatenate((x, y), axis = 1).reshape(212,)
          #  parameters=c.tolist()
            parameter_list.append(parameters)
            name_list.append(name)
    return name_list,parameter_list

def change_pose(filepath, target_path, txt_root, save_root, name):

    Dir = os.listdir(filepath)

    for i in range(len(Dir)):
    
        f_name = Dir[i].split('.')[0]
        emotion = f_name.split('_')[1]
        fname =  f_name.split('_')[-1]
        
        target_pose = np.load(os.path.join(target_path,name+'.npy'))
 #       target_pose = np.load(os.path.join(target_path,name+'_'+emotion+'_'+fname+'.npy')
        target_pose = np.vstack((target_pose,np.flip(target_pose,0)))
        index = 0
 #       txt_path = os.path.join(txt_root,emotion+'/'+fname+'/'+fname+'.txt'
        txt_path = os.path.join(txt_root, name, name+'.txt') #f_name+'/'+f_name+'.txt'
        names,parameters = default_parameter_reader(txt_path)
        para = parameters[-1::-1]
        parameters = parameters+para
        save_path = os.path.join(save_root,f_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        first = parameters[0]
        first = np.array(first).reshape(106,2) 
    
        np.savetxt(os.path.join(save_path , '{:06d}'.format(0)+'.txt'), first, fmt='%d', delimiter=',')
        second = parameters[1]
        second = np.array(second).reshape(106,2) 
    
        np.savetxt(os.path.join(save_path , '{:06d}'.format(1)+'.txt'), second, fmt='%d', delimiter=',')
        number = 0
        res_all = []
 
        for line in open(os.path.join(filepath, f_name+'.txt')).readlines():
            info = line.split()
            i_name = info[-1]
            video_name = i_name.split('.')[0]
         
            if(int(video_name)==number):
                
                print('IMG: {}'.format(video_name))                
                
                number += 1
                shape = torch.tensor([float(it) for it in info[:199]]).unsqueeze(0).float().cuda()
                exp = torch.tensor([float(it) for it in info[199:199 + 29]]).unsqueeze(0).float().cuda()

                if index < len(target_pose)-1 :
                
                    pose_a = target_pose[index+2].tolist()
                else:
                    pose_a = target_pose[len(target_pose)-1].tolist()
                pose_b = [float(it) for it in info[199 + 29:199 + 29 + 6]]
                pose = pose_a[:3] + pose_b[3:]

                pose = torch.tensor(pose).unsqueeze(0).float().cuda()
                
                ldmk_3d = exp2ldmk(shape, exp, pose)[:, ldmk_idx].transpose(0, 1) \
                    .contiguous().cpu().view(1, -1)[0].tolist()
                pose_3d = pose.contiguous().cpu().tolist()[0]
                  # pt_x, pt_y = ldmk3d_2d(pose, torch.tensor(ldmk_3d).view(106, 3).transpose(0, 1).cuda())
                ldmk_3d = [str(it) for it in ldmk_3d]
                pose_3d = [str(it) for it in pose_3d]
        
                ldmk_de = exp2ldmk(shape, exp, pose)[:, ldmk_idx].contiguous()
                pose_de = pose.contiguous()
                raw_x,raw_y = ldmk3d_2d(pose_de,ldmk_de)
       
                x = raw_x.cpu().tolist()
                x = np.array(x).reshape(106,1)
     
                y = raw_y.cpu().tolist()
                y = np.array(y).reshape(106,1)
    
                source = np.concatenate((x, y), axis = 1)

                target = parameters[index+2]
                target = np.array(target).reshape(106,2)
                target[:,0] = target[:,0] -420
                s = np.vstack((source[104:,:],source[46,:],source[33,:],source[42,:]))
                t = np.vstack((target[104:,:],target[46,:],target[33,:],target[42,:]))
                M=_umeyama(s, t, True)
                c=np.ones((106,1))
                b=np.hstack((source,c))
                res=np.dot(M,b.T)
                res=res[:2,:]
                res=res.T
                res[:,0] = res[:,0]+420
                res_all.append(res)
                
                index += 1

        one_euro_filter = OneEuroFilter(mincutoff=0.01, beta=0.7, dcutoff=1.0, freq=100)
        new = np.array(res_all).reshape(-1,212)
          
        for j in range(len(new)):
            new[j]=one_euro_filter.process(new[j])

        for k in range(len(new)):
      
            tar = new[k].reshape(-1,2)
            save_name = os.path.join(save_path , '{:06d}'.format(k+2)+'.txt')
            np.savetxt(save_name, tar, fmt='%d', delimiter=',')
         #   print('TXT: {}'.format(k+2))  
            
        print(i,f_name,index)

def prepare_image(save_root,image_root, out_root, name):
   
    Dir = os.listdir(save_root)
    for i in range(len(Dir)):
        n = Dir[i]
        emotion = n.split('_')[1]
        fname =  n.split('_')[-1]
  #      image_path = os.path.join(image_root,emotion+'/'+fname+'/image/')
        image_path = os.path.join(image_root,name+'/image/') #Dir[i]+'/image/'
        key_path = os.path.join(save_root,n)
        out_path = os.path.join(out_root,n)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        length = len(os.listdir(key_path))
        for j in range(length):
   
            file_length = len(os.listdir(image_path))
            if j < file_length:
                shutil.copyfile(image_path+str(j)+'.jpg',out_path+'/'+'{:06d}'.format(j)+'.jpg')
            else:
                shutil.copyfile(image_path+str(2*file_length-1-j)+'.jpg',out_path+'/'+'{:06d}'.format(j)+'.jpg')
        print(i, j)


# num_shape = 199
# num_exp = 29
# num_pose = 6

if __name__ == "__main__":
    with open("./ldmk_idx.txt", 'r') as f:
        line = f.readline()
    info = line.strip().split()
    ldmk_idx = [int(it) for it in info]
    params = {}
    exp2ldmk = Exp2Ldmk(params)
    #get landmark txt for vid2vid
    name = 'M003' #'M030'
    filepath = 'data/'+name+'/3DMM/test_results/' 
    target_path = 'data/'+name+'/background/'#'data/'+name+'/3DMM/'+name+'_test_pose/'
    txt_root = 'data/'+name+'/background/'#'/3DMM/3DMM/'
    save_root = 'result/test_keypoints_'+name+'_pose/'
    change_pose(filepath, target_path, txt_root, save_root, name)

    #get image for vid2vid
    image_root = txt_root
    out_root = 'result/test_img_'+name+'_pose/'
    prepare_image(save_root,image_root, out_root, name)






