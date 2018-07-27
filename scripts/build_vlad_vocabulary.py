'''
build_vlad_vocabulary.py

author  : cfeng, schaturvedi
created : 4/22/18 9:56 PM
'''

import os
import sys
import glob
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt

import cv2
from sklearn.cluster import MiniBatchKMeans

def read_list(input_list_file, skip_count):
    lines = open(input_list_file).readlines()[1:]
    if skip_count != 0:
        lines = [lines[i].split(' ')[0].strip() for i in xrange(0, len(lines), skip_count)]
    else:
        lines = [lines[i].split(' ')[0].strip() for i in xrange(0, len(lines))]
    assert(len(lines)>0)
    return lines

def load_casenet_result(img_h, img_w, result_fmt, K_class, idx_base):
    prob = np.zeros((img_h, img_w, K_class), dtype=np.float32)
    for k in xrange(K_class):
        prob_k = cv2.imread(result_fmt%(k+idx_base), cv2.IMREAD_GRAYSCALE)
        prob[:,:,k] = prob_k.astype(np.float32) / 255.
    return prob

def describeSIFT(image):
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(image,None)
    return kp,des

def main(args):

    KM = MiniBatchKMeans(
        n_clusters=args.vlad_feature_dimension,
        max_iter=args.kmean_max_iter,
        n_init=20,
        verbose=True,
        compute_labels=False
    )


    #parse input list
    lines = read_list(args.input_list,args.skip_count)
    img_example = cv2.imread(
        os.path.join(args.src ,lines[0])
    )
    img_example = cv2.cvtColor(img_example, cv2.COLOR_BGR2RGB)
    img_h, img_w, img_dim = img_example.shape
    
    print "Height of input images =", img_h
    print "width of input images =", img_w
    
    if args.feature_type.lower() == 'casenet':
        ii, jj = np.meshgrid(xrange(img_h), xrange(img_w), indexing='ij')
        ii = ii[...,np.newaxis]/(img_h+0.0) #img_h x img_w => img_h x img_w x 1
        jj = jj[...,np.newaxis]/(img_w+0.0)

        all_features = None
        for l in lines:
            prob = load_casenet_result(
                img_h, img_w,
                result_fmt=os.path.join(args.result_root,'class_%d',l),
                K_class=args.result_classes,
                idx_base=0
            )

            #Remove given set of classes from casenet feature list
            if args.removed_class:
                prob = np.delete(prob,args.removed_class,axis=2)
            
            feat = np.concatenate((ii,jj,prob),axis=2).astype(np.float32)
            feat = feat.reshape(np.prod(feat.shape[:2]), feat.shape[-1])
            mask = ((prob>args.thresh).sum(axis=2))>0
            mask = mask.reshape(-1)
            feat = feat[mask,...]
            feat[:,0] *= float(1 - args.alpha)
            feat[:,1] *= float(1 - args.alpha)
            feat[:,2:] *= args.alpha
            print('#features for {:s} = {:d}'.format(l, feat.shape[0]))
            KM.partial_fit(feat)
    else:
        xy_weight = float(1 - args.alpha)
        for l in lines:
            im=cv2.imread(os.path.join(args.src,l))
            kp,des = describeSIFT(im)
            desc = np.empty((des.shape[0],130))
            for i,keypoint in enumerate(kp):
                desc[i] = np.hstack((des[i],keypoint.pt))
                desc[i][-1] /= img_h
                desc[i][-2] /= img_w
                desc[i][0:128] /= 255
                desc[i][-1] *= xy_weight
                desc[i][-2] *= xy_weight
                desc[i][0:128] = list(np.asarray(desc[i][0:128])*args.alpha)
            KM.partial_fit(desc)
            print(l,desc.shape)

    np.savez_compressed(args.vocabulary_file_prefix+'.npz', vocabulary=KM.cluster_centers_)
    print('saved: '+args.vocabulary_file_prefix+'.npz')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(sys.argv[0])

    # By default, the script uses SIFT features.
    parser.add_argument('-f','--feature_type', type=str, default='sift',
                    help="Feature type used for localization - casenet or sift")

    # In our experiments, SIFT performed best with cluster=32 and Casenet was best with cluster=64.
    parser.add_argument('-c', '--vlad_feature_dimension', type=int, default='32',
                    help="Number of clusters for VLAD")

    # Alpha gives weight to the features (Casenet/SIFT vs XY).  XY will have weight = (1-alpha)
    # alpha=0.5 means equal weight to both. In our experiments, alpha=0.1 gave the best results.
    parser.add_argument('-a', '--alpha', type=float, default='0.5',
                    help="Weight given to the features of feature_type, xy will have weight = (1-alpha)")

    # To reduce the execution time, we skip video frames. It did not make much difference in the performance
    # as we skip according to the frame rate of videos. For e.g. KAIST dataset has frame rate = 10, so we 
    # skip 5 frames while creating vocabulary as it does not cause much loss of information.
    parser.add_argument('-s', '--skip_count', type=int, default='0',
                    help="Number of frames to skip while training and testing VLAD")

    ### Required arguments

    # The input list must have the data in following space separated format:
    # ImageFile X Y
    # AM09_000000.png 36.37261637 127.3641256
    # X and Y are gps coordinates of the frame in meters.
    parser.add_argument('-i', '--input_list', type=str,required=True,
                    help="Absolute path of the training images list with gps information")
    parser.add_argument('-src', '--src', type=str,required=True,
                    help="Directory containing original source training images")

    ### Casenet specific arguments
    parser.add_argument('-r', '--result_root', type=str,default=None,
                    help="Directory containing casenet features of training images")
    parser.add_argument('-t', '--thresh', type=float, default='0.5',
                    help="Probability threshold value to use casenet features")
    parser.add_argument('-rc', '--result_classes', type=int, default=19,
                    help="Number of casenet classes")
    parser.add_argument('-crf', '--removed_class',type=str, default=None,
                    help="Comma seperated list of casenet class ids (Refer readme for class id) that should be removed during the experiment")

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    args.vocabulary_file_prefix = 'vocabulary_f={:s}_k={:d}_t={:.1f}_alpha={:.1f}'.format(
        args.feature_type,
        args.vlad_feature_dimension,
        args.thresh,
        args.alpha
    )
    print "Vocabulary file will be created by name:",args.vocabulary_file_prefix+'.npz'
    args.all_feature_file = args.vocabulary_file_prefix+'.all_features.npz'
    args.use_minibatch_kmeans = True
    args.kmean_max_iter = 10000

    if args.feature_type.lower() == 'casenet':
        if args.result_root is None:
            parser.error("casenet requires valid value of --result_root.")
        else:
            assert(os.path.exists(args.result_root))
        if args.removed_class:
            args.removed_class = args.removed_class.strip().split(',')
            if any( int(x) < 0 or int(x) > 18 for x in args.removed_class):
                parser.error("--removed_class must have comma seperated values between 0 to 18.")
            print "CaseNet classes to be removed from this experiment =", args.removed_class

    assert(os.path.exists(args.input_list))

    main(args)
