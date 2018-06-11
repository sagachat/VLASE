'''
compute_vlad_descriptor.py

author  : schaturvedi
created : 5/8/18 12:21 AM
'''

import os
import sys
import glob
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt

import cv2

from sklearn.cluster import KMeans

from build_vlad_vocabulary import read_list,load_casenet_result,describeSIFT

class XYCASENetVLAD(object):
    def __init__(self,
                 VOCAB,
                 img_h, img_w,
                 thresh,
                 do_SSR=True):
        self.VOCAB = VOCAB
        self.nclusters, ndim = self.VOCAB.shape
        assert(ndim==21)

        self.KM = KMeans(self.nclusters, copy_x=False)
        self.KM.cluster_centers_ = self.VOCAB

        ii, jj = np.meshgrid(xrange(img_h), xrange(img_w), indexing='ij')
        ii = ii[...,np.newaxis]/(img_h+0.0) #img_h x img_w => img_h x img_w x 1
        jj = jj[...,np.newaxis]/(img_w+0.0)
        self.ii = ii
        self.jj = jj
        self.img_h = img_h
        self.img_w = img_w

        self.thresh = thresh
        self.do_SSR = do_SSR


    def _convert_CASENet_prob_to_features(self, prob, args):
        img_h, img_w, ndim = prob.shape
        assert(img_h==self.img_h and img_w==self.img_w and ndim==19)
        feat = np.concatenate((self.ii,self.jj,prob),axis=2).astype(np.float32)
        feat = feat.reshape(np.prod(feat.shape[:2]), feat.shape[-1])
        mask = ((prob>self.thresh).sum(axis=2))>0
        mask = mask.reshape(-1)
        feat = feat[mask,...]
        feat[:,0] *= float(1 - args.alpha)
        feat[:,1] *= float(1 - args.alpha)
        feat[:,2:] *= args.alpha
        #print('#features = {:s}'.format( feat.shape))
        #print max(feat[:,0]),max(feat[:,1]),max(feat[:,2]),max(feat[:,-1])
        return feat


    def compute(self, prob, args):
        feat = self._convert_CASENet_prob_to_features(prob,args)
        nsamples, ndim = feat.shape

        vlad = np.zeros((self.nclusters, ndim), dtype=np.float32)
        cidx = self.KM.predict(feat)

        # compute unnormalized VLAD
        for c in xrange(self.nclusters):
            F = feat[cidx==c]
            C = np.tile(self.VOCAB[c], (F.shape[0],1))
            vlad[c] = (F - C).sum(axis=0)
        vlad = vlad.reshape(-1) # kxd -> kdx1

        if self.do_SSR:
            vlad = np.sign(vlad)*np.sqrt(np.abs(vlad))

        #L2-normalization
        vlad_norm = np.linalg.norm(vlad)
        if vlad_norm>1e-6:
            vlad = vlad/vlad_norm
        else:
            vlad[...]=0

        return vlad.astype(np.float32)



class VLAD(object):
    def __init__(self,
                 VOCAB):
        self.VOCAB = VOCAB
        self.nclusters, ndim = self.VOCAB.shape

        self.KM = KMeans(self.nclusters, copy_x=False)
        self.KM.cluster_centers_ = self.VOCAB

    def compute(self, feat):
        nsamples, ndim = feat.shape

        vlad = np.zeros((self.nclusters, ndim), dtype=np.float32)
        cidx = self.KM.predict(feat)

        for i in xrange(self.nclusters):
            # if there is at least one descriptor in that cluster
            if np.sum(cidx==i)>0:
                # add the diferences
                vlad[i]=np.sum(feat[cidx==i,:]-self.VOCAB[i],axis=0)

        vlad = vlad.flatten()

        vlad = np.sign(vlad)*np.sqrt(np.abs(vlad))

        #L2-normalization
        vlad_norm = np.linalg.norm(vlad)
        if vlad_norm>1e-6:
            vlad = vlad/vlad_norm
        else:
            vlad[...]=0

        return vlad.astype(np.float32)


def main(args):

    VOCAB = np.load(args.vocab_file)['vocabulary']

    #parse input list
    lines = read_list(args.input_list,args.skip_count)
    img_example = cv2.imread(
        os.path.join(args.src ,lines[0])
    )
    img_example = cv2.cvtColor(img_example, cv2.COLOR_BGR2RGB)
    img_h, img_w, img_dim = img_example.shape
    
    print "Height of input images =", img_h
    print "width of input images =", img_w
    
    name2vlad = {}

    if args.feature_type.lower() == 'casenet':
        _VLAD = XYCASENetVLAD(
            VOCAB=VOCAB,
            img_h=img_h, img_w=img_w,
            thresh=args.thresh
        )

        for l in lines:
            prob = load_casenet_result(
                    img_h, img_w,
                    result_fmt=os.path.join(args.result_root,'class_%d',l),
                    K_class=args.result_classes,
                    idx_base=0)
            #Remove given set of classes from casenet feature list
            if args.removed_class:
                prob = np.delete(prob,args.removed_class,axis=2)
            vlad = _VLAD.compute(prob,args)
            name2vlad[l] = vlad
            print(l.split('_')[0]+'/'+l)
    else:
        _VLAD = VLAD(VOCAB=VOCAB)
        xy_weight = float(1 - args.alpha)
        for l in lines:
            im=cv2.imread(os.path.join(args.src,l))
            kp,des = describeSIFT(im)
            desc = np.empty((des.shape[0],130))
            for i,keypoint in enumerate(kp):
                desc[i] = np.hstack((des[i],keypoint.pt))
                desc[i][-1] /= img_w
                desc[i][-2] /= img_h
                desc[i][0:128] /= 255
                desc[i][-1] *= xy_weight
                desc[i][-2] *= xy_weight
                desc[i][0:128] = list(np.asarray(desc[i][0:128])*args.alpha)
            vlad = _VLAD.compute(desc)
            name2vlad[l] = vlad
            print(l,desc.shape)

    np.savez_compressed(args.output, name2vlad=name2vlad)
    print('saved: '+args.output)



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

    # To reduce the execution time, we skip video frames. It does not make much difference in the performance
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
                    help="Absolute path of the images list with gps information, this list must have both training and testing data.")
    parser.add_argument('-src', '--src', type=str,required=True,
                    help="Directory containing original source images, this directory must have both training and testing data.")

    ### Casenet specific arguments
    parser.add_argument('-r', '--result_root', type=str,default=None,
                    help="Directory containing casenet features of both training and testing images")
    parser.add_argument('-t', '--thresh', type=float, default='0.5',
                    help="Probability threshold value to use casenet features")
    parser.add_argument('-rc', '--result_classes', type=int, default=19,
                    help="Number of casenet classes")
    parser.add_argument('-crf', '--removed_class',type=str, default=None,
                    help="Comma seperated list of casenet class ids (Refer readme for class id) that should be removed during the experiment")

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    args.vocab_file = 'vocabulary_f={:s}_k={:d}_t={:.1f}_alpha={:.1f}.npz'.format(
        args.feature_type,
        args.vlad_feature_dimension,
        args.thresh,
        args.alpha
    )
    args.output = args.vocab_file.strip('.npz')+'.vlad.npz'
    print "Vocabulary file :",args.vocab_file,", Output VLAD file:",args.output
    
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