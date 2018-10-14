from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import numpy as np
import pickle

_RANDOM_SEED = 0
random.seed(_RANDOM_SEED)

def _get_folder_path(dataset_dir, split_name):
    if split_name == 'train':
        folder_path = os.path.join(dataset_dir, 'filted_up_train')
    elif split_name == 'train_flip':
        folder_path = os.path.join(dataset_dir, 'filted_up_train_flip')
    elif split_name == 'test':
        folder_path = os.path.join(dataset_dir, 'filted_up_test')
    assert os.path.isdir(folder_path)
    return folder_path


def _get_image_file_list(dataset_dir, split_name):
    folder_path = _get_folder_path(dataset_dir, split_name)
    if split_name == 'train' or split_name == 'train_flip':
        filelist = sorted(os.listdir(folder_path))
        filelist = sorted(filelist)
    elif split_name == 'test':
        filelist = sorted(os.listdir(folder_path))
        
    valid_filelist = []
    for i in range(0, len(filelist)):
        if filelist[i].endswith('.jpg') or filelist[i].endswith('.png'):
            valid_filelist.append(filelist[i])

    return valid_filelist


def _get_train_all_pn_pairs(dataset_dir, out_dir, split_name='train', augment_ratio=1):
    """Returns a list of pair image filenames.
    Args:
        dataset_dir: A directory containing person images.
    Returns:
        p_pairs: A list of positive pairs.
        n_pairs: A list of negative pairs.
    """
    assert split_name in {'train', 'train_flip', 'test'}
    if split_name == 'train_flip':
        p_pairs_path = os.path.join(out_dir, 'p_pairs_train_flip.p')
        n_pairs_path = os.path.join(out_dir, 'n_pairs_train_flip.p')
    else:
        p_pairs_path = os.path.join(out_dir, 'p_pairs_' + split_name.split('_')[0] + '.p')
        n_pairs_path = os.path.join(out_dir, 'n_pairs_' + split_name.split('_')[0] + '.p')
    
    if os.path.exists(p_pairs_path):
        with open(p_pairs_path, 'r') as f:
            p_pairs = pickle.load(f)
        with open(n_pairs_path, 'r') as f:
            n_pairs = pickle.load(f)
    else:
        filelist = _get_image_file_list(dataset_dir, split_name)
        filenames = []
        p_pairs = []
        n_pairs = []
        
        for i in range(0, len(filelist)):
            names = filelist[i].split('_')
            id_i = names[0]
            for j in range(i+1, len(filelist)):
                names = filelist[j].split('_')
                id_j = names[0]
                if id_j == id_i:
                    p_pairs.append([filelist[i], filelist[j]])
                    p_pairs.append([filelist[j], filelist[i]])  # if two streams share the same weights, no need switch
                    if len(p_pairs) % 100000 == 0:
                        print(len(p_pairs))
                elif j % 2000 == 0 and id_j != id_i:  # limit the neg pairs to 1/40, otherwise it cost too much time
                    n_pairs.append([filelist[i], filelist[j]])
                    if len(n_pairs) % 100000 == 0:
                        print(len(n_pairs))

        print('repeat positive pairs augment_ratio times and cut down negative pairs to balance data ......')
        p_pairs = p_pairs * augment_ratio  
        random.shuffle(n_pairs)
        n_pairs = n_pairs[:len(p_pairs)]
        print('p_pairs length:%d' % len(p_pairs))
        print('n_pairs length:%d' % len(n_pairs))
        print('save p_pairs and n_pairs ......')
        with open(p_pairs_path, 'wb') as f:
            pickle.dump(p_pairs, f)
        with open(n_pairs_path, 'wb') as f:
            pickle.dump(n_pairs, f)

    print('_get_train_all_pn_pairs finish ......')
    print('p_pairs length:%d' % len(p_pairs))
    print('n_pairs length:%d' % len(n_pairs))

    print('save pn_pairs_num ......')
    pn_pairs_num = len(p_pairs) + len(n_pairs)

    if split_name=='train_flip':
        fpath = os.path.join(out_dir, 'pn_pairs_num_train_flip.p')
    else:
        fpath = os.path.join(out_dir, 'pn_pairs_num_' + split_name.split('_')[0] + '.p')
    with open(fpath, 'wb') as f:
        pickle.dump(pn_pairs_num, f)

    return p_pairs, n_pairs


def run_one_pair_rec(dataset_dir, out_dir, split_name): 
    
    if split_name.lower()=='train':
        #================ Prepare training set ================
        pose_peak_path = os.path.join(dataset_dir, 'PoseFiltered', 'all_peaks_dic_DeepFashion.p')
        pose_sub_path = os.path.join(dataset_dir, 'PoseFiltered', 'subsets_dic_DeepFashion.p')
        pose_peak_path_flip = os.path.join(dataset_dir, 'PoseFiltered', 'all_peaks_dic_DeepFashion_Flip.p')
        pose_sub_path_flip = os.path.join(dataset_dir, 'PoseFiltered', 'subsets_dic_DeepFashion_Flip.p')

        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                    split_name=split_name,
                                                    augment_ratio=1)
        p_labels = [1]*len(p_pairs) 
        n_labels = [0]*len(n_pairs) 
        pairs = p_pairs
        labels = p_labels
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        split_name_flip='train_flip'
        p_pairs_flip, n_pairs_flip = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                    split_name=split_name_flip,
                                                    augment_ratio=1)
        p_labels_flip = [1]*len(p_pairs_flip)
        n_labels_flip = [0]*len(n_pairs_flip)
        pairs_flip = p_pairs_flip
        labels_flip = p_labels_flip
        combined = list(zip(pairs_flip, labels_flip))
        random.shuffle(combined)
        pairs_flip[:], labels_flip[:] = zip(*combined)

        print('\nTrain convert Finished !')

    if split_name.lower()=='test':
        # ================ Prepare test set ================
        pose_peak_path = os.path.join(dataset_dir, 'PoseFiltered', 'all_peaks_dic_DeepFashion.p')
        pose_sub_path = os.path.join(dataset_dir, 'PoseFiltered', 'subsets_dic_DeepFashion.p')

        p_pairs, n_pairs = _get_train_all_pn_pairs(dataset_dir, out_dir,
                                                  split_name=split_name,
                                                  augment_ratio=1)
        p_labels = [1]*len(p_pairs)
        n_labels = [0]*len(n_pairs)
        pairs = p_pairs
        labels = p_labels
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs[:], labels[:] = zip(*combined)

        ## Test will not use flip
        split_name_flip = None
        pairs_flip = None
        labels_flip = None

        print('\nTest samples convert Finished !')


if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    split_name = sys.argv[2]  
    out_dir = os.path.join(dataset_dir, 'DF_' + split_name.replace('_flip', '') + '_data')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    run_one_pair_rec(dataset_dir, out_dir, split_name)
