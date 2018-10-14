import math
import os
import random
import sys

import torch
import torch.utils.data

import numpy as np
import scipy.io
import scipy.stats
import skimage.morphology
from scipy import misc
from skimage.morphology import square, dilation, erosion

def _get_train_all_p_pairs(out_dir, split_name='train'):
    assert split_name in {'train', 'train_flip', 'test'}
    if split_name == 'train_flip':
        p_pairs_path = os.path.join(out_dir, 'p_pairs_train_flip.p')
    else:
        p_pairs_path = os.path.join(out_dir, 'p_pairs_' + split_name.split('_')[0] + '.p')
        
    if os.path.exists(p_pairs_path):
        with open(p_pairs_path, 'rb') as f:
            p_pairs = pickle.load(f)
            
    print('_get_train_all_pn_pairs finish ...')
    print('p_pairs length:%d' % len(p_pairs))
    
    return p_pairs


def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2 + j**2))
            if r+i >= 0 and r+i < height and c+j >= 0 and c+j < width:
                if 'Solid' == mode and distance <= radius:
                    indices.append([r+i, c+j, k])

    return indices


def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0!=len(p):
            r = p[0][1]
            c = p[0][0]
            ind = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)
            
    shape = [height, width, channel]
    return indices, shape


def _oneDimSparsePose(indices, shape):
    ind_onedim = []
    for ind in indices:
        idx = ind[0]*shape[2]*shape[1] + ind[1]*shape[2] + ind[2]
        ind_onedim.append(idx)
    shape = np.prod(shape)
    return ind_onedim, shape


def _sparse2dense(indices, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = 1
    return dense


def _getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                         [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                         [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    indices = []
    for limb in limbSeq:
        p0 = peaks[limb[0] -1]
        p1 = peaks[limb[1] -1]
        if 0!=len(p0) and 0!=len(p1):
            r0 = p0[0][1]
            c0 = p0[0][0]
            r1 = p1[0][1]
            c1 = p1[0][0]
            ind  = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            ind = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)
            
            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            if sampleN > 1:
                for i in range(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)
                    
    shape = [height, width, 1]
    
    dense = np.squeeze(_sparse2dense(indices, shape))
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense


def _get_valid_peaks(all_peaks, subsets):
    try:
        subsets = subsets.tolist()
        valid_idx = -1
        valid_score = -1
        for i, subset in enumerate(subsets):
            score = subset[-2]   
            if score > valid_score:
                valid_idx = i
                valid_score = score
        if valid_idx >= 0:
            return all_peaks
        else:
            return None
    except:
        return None


def _format_data(folder_path, pairs, i, all_peaks_dic, subsets_dic):
    # Read the filename:
    img_path_0 = os.path.join(folder_path, pairs[i][0])
    img_path_1 = os.path.join(folder_path, pairs[i][1])
    image_raw_0 = misc.imread(img_path_0)
    image_raw_1 = misc.imread(img_path_1)
    height, width = image_raw_0.shape[1], image_raw_0.shape[0]

    ########################## Pose 16x8 & Pose coodinate (for 128x64(Solid) 128x64(Gaussian))##########################
    if (all_peaks_dic is not None) and (pairs[i][0] in all_peaks_dic) and (pairs[i][1] in all_peaks_dic):
        ## Pose 1
        peaks = _get_valid_peaks(all_peaks_dic[pairs[i][1]], subsets_dic[pairs[i][1]])
        indices_r4_1, shape = _getSparsePose(peaks, height, width, 18, radius=4, mode='Solid')
        indices_r4_1, shape_1 = _oneDimSparsePose(indices_r4_1, shape)
        pose_mask_r4_1 = _getPoseMask(peaks, height, width, radius=4, mode='Solid')
    else:
        return None

    image_raw_0 = np.reshape(image_raw_0, (height, width, 3))
    image_raw_0 = image_raw_0.astype('float32')
    image_raw_1 = np.reshape(image_raw_1, (height, width, 3))
    image_raw_1 = image_raw_1.astype('float32')

    mask_1 = np.reshape(pose_mask_r4_1, (height, width, 1))
    mask_1 = mask_1.astype('float32')

    indices_r4_1 = np.array(indices_r4_1).astype(np.int64).flatten().tolist()
    indices_r4_1_dense = np.zeros((shape_1))
    indices_r4_1_dense[indices_r4_1] = 1
    indices_r4_1 = np.reshape(indices_r4_1_dense, (height, width, 18))
    pose_1 = indices_r4_1.astype('float32')

    image_0 = (image_raw_0 - 127.5) / 127.5
    image_1 = (image_raw_1 - 127.5) / 127.5
    pose_1 = pose_1 * 2 - 1
    
    image_0 = torch.from_numpy(np.transpose(image_0, (2, 0, 1)))
    image_1 = torch.from_numpy(np.transpose(image_1, (2, 0, 1)))
    mask_1 = torch.from_numpy(np.transpose(mask_1, (2, 0, 1)))
    pose_1 = torch.from_numpy(np.transpose(pose_1, (2, 0, 1)))
    
    return [image_0, image_1, pose_1, mask_1]


class PoseDataset(torch.utils.data.Dataset):
    """Pose dataset."""    
    def __init__(self, out_dir, folder_path, folder_path_flip, pose_peak_path, pose_sub_path, pose_peak_path_flip, pose_sub_path_flip):
        self.folder_path = folder_path
        self.folder_path_flip = folder_path_flip
        self.p_pairs = _get_train_all_p_pairs(out_dir)
        self.p_pairs_flip = _get_train_all_p_pairs(out_dir, 'train_flip')
        self.length = len(self.p_pairs) + len(self.p_pairs_flip)

        self.all_peaks_dic = None
        self.subsets_dic = None
        self.all_peaks_dic_flip = None
        self.subsets_dic_flip = None
        
        with open(pose_peak_path, 'rb') as f:
            self.all_peaks_dic = pickle.load(f, encoding='latin1')
        with open(pose_sub_path, 'rb') as f:
            self.subsets_dic = pickle.load(f, encoding='latin1')
        
        with open(pose_peak_path_flip, 'rb') as f:
            self.all_peaks_dic_flip = pickle.load(f, encoding='latin1')
        with open(pose_sub_path_flip, 'rb') as f:
            self.subsets_dic_flip = pickle.load(f, encoding='latin1')
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        while True:
            USE_FLIP = index >= len(self.p_pairs)
            if USE_FLIP:
                example = _format_data(self.folder_path_flip, p_pairs_flip, index - len(self.p_pairs), self.all_peaks_dic_flip, self.subsets_dic_flip)
                if example:
                    return example
                index = (index + 1) % length
            else:
                example = _format_data(self.folder_path, p_pairs, index, self.all_peaks_dic, self.subsets_dic)
                if example:
                    return example
                index = (index + 1) % length


    def get_loader(dataset_dir, batch_size):
        pose_dataset = PoseDataset(os.path.join(dataset_dir, 'DF_train_data'),
                                   os.path.join(dataset_dir, 'filted_up_train'),
                                   os.path.join(dataset_dir, 'filted_up_train_flip'),
                                   os.path.join(dataset_dir, 'PoseFiltered', 'all_peaks_dic_DeepFashion.p'),
                                   os.path.join(dataset_dir, 'PoseFiltered', 'subsets_dic_DeepFashion.p'),
                                   os.path.join(dataset_dir, 'PoseFiltered', 'all_peaks_dic_DeepFashion_Flip.p'),
                                   os.path.join(dataset_dir, 'PoseFiltered', 'subsets_dic_DeepFashion_Flip.p'))
        pose_loader = torch.utils.data.DataLoader(pose_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
