'''
vlad_place_recognition.py

author  : cfeng, schaturvedi
created : 6/7/18 1:15 AM
'''

import os
import sys
import glob
import time
import argparse
import math
import numpy as np
from matplotlib import pyplot as plt

import cv2

def read_records(input_records_file):
    lines = open(input_records_file).readlines()[1:]

    records = [
        l.strip().split(' ')
        for l in lines
    ]

    records = np.array(
        [(r[0], float(r[1]), float(r[2]))
         for r in records],
        dtype=[('name', object) ,('x', float), ('y', float)]
    )

    return records

def main(args):
    

    #load data    
    train_records = read_records(args.train_list)
    train_names = [r[0] for r in train_records]
    test_records = read_records(args.test_list)
    test_names = [r[0] for r in test_records]
    name2vlad = np.load(args.vlad_features)['name2vlad'].item()
    #print(len(train_records),len(test_records),len(train_names),len(test_names))

    def get_records():

        map_idx=[]
        loc_idx=[]
        map_vlad=[]
        loc_vlad=[]
        for ith in xrange(len(train_records)):
            ith_name = train_names[ith]
            #skip images whose vlad are not computed
            if not name2vlad.has_key(ith_name):
                continue
            map_idx.append(ith)
            map_vlad.append(name2vlad[ith_name])
            
        for ith in xrange(len(test_records)):
            ith_name = test_names[ith]
            #skip images whose vlad are not computed
            if not name2vlad.has_key(ith_name):
                continue
            loc_idx.append(ith)
            loc_vlad.append(name2vlad[ith_name])
        #print(len(map_idx),len(loc_idx),len(map_vlad),len(loc_vlad))
        return map_idx, loc_idx, map_vlad, loc_vlad

    map_idx, loc_idx, map_vlad, loc_vlad = get_records()
    assert(len(map_idx)>=args.top_k)
    map_vlad_mat = np.array(map_vlad)
    loc_pairs = zip(loc_idx, loc_vlad)

    def distance_for_global_coordinates(lat1, lng1, lat2, lng2):
        #return distance as meter between global latitude and longitude. if you want km distance, remove "* 1000"
        radius = 6371 * 1000 

        dLat = (lat2-lat1) * math.pi / 180
        dLng = (lng2-lng1) * math.pi / 180

        lat1 = lat1 * math.pi / 180
        lat2 = lat2 * math.pi / 180

        val = math.sin(dLat/2) * math.sin(dLat/2) + math.sin(dLng/2) * math.sin(dLng/2) * math.cos(lat1) * math.cos(lat2)    
        ang = 2 * math.atan2(math.sqrt(val), math.sqrt(1-val))
        return radius * ang
    
    def whether_should_found(r_query):
        '''check if query record r_query falls within acceptance distance of any record in the map'''
        if args.gps_type.lower() == 'global':
            d = []
            for ith in map_idx:
                lat1 = train_records[ith]['x']
                lng1 = train_records[ith]['y']
                lat2 = r_query['x']
                lng2 = r_query['y']
                dist = distance_for_global_coordinates(lat1, lng1, lat2, lng2)
                d.append(dist)
            d = np.array(d)
            return np.any(d<=args.acceptance_distance_thresh)
        else:
            xy_map = np.array([(train_records[ith]['x'],train_records[ith]['y']) for ith in map_idx])
            xy_query = np.tile(np.array([r_query['x'], r_query['y']]), (len(map_idx), 1))
            dxy = xy_map - xy_query
            d = np.sqrt( (dxy*dxy).sum(axis=1) )
            return np.any(d<=args.acceptance_distance_thresh)



    #localize all images with vlad but not used for mapping/building vocabulary
    total_found = np.zeros((args.top_k,), dtype=int)
    total_should_found = 0
    for ith, ith_vlad in loc_pairs:
        ith_name = test_names[ith]
        ith_rec = test_records[ith]
        ith_vlad = name2vlad[ith_name]

        similarities = map_vlad_mat.dot(ith_vlad)
        results = np.argsort(similarities)[::-1] #descending order
        results = results[:args.top_k]

        total_should_found += whether_should_found(ith_rec)
        found = np.zeros((args.top_k,), dtype=bool)
        for k in xrange(args.top_k):
            if similarities[results[k]]<args.similarity_thresh:
                continue
            jth = map_idx[results[k]]
            jth_rec = train_records[jth]
            if args.gps_type.lower() == 'global':
                dij = distance_for_global_coordinates(jth_rec['x'],jth_rec['y'],ith_rec['x'],ith_rec['y'])
            else:
                dij = np.sqrt((ith_rec['x']-jth_rec['x'])**2+(ith_rec['y']-jth_rec['y'])**2)
            found[k] = dij<=args.acceptance_distance_thresh
        top_k_found = np.cumsum(found)>0

        total_found += top_k_found
        #print('processed {}'.format(ith_name))


    #report results
    print('for acceptance distance={:.1f}m:'.format(args.acceptance_distance_thresh))
    for k in xrange(args.top_k):
        print('\ttop-{:d}-accuracy={:d}/{:d}={:.2f}%'.format(
            k+1,
            total_found[k],
            total_should_found,
            total_found[k]*100.0/total_should_found))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    ### Required arguments

    # The input lists must have the data in following space separated format:
    # ImageFile X Y
    # AM09_000000.png 36.37261637 127.3641256
    # X and Y are gps coordinates of the frame in meters.
    parser.add_argument('-tr', '--train_list', type=str,required=True,
                    help="Absolute path of the training images list with gps information. This is the same list used for building vocabulary.")
    parser.add_argument('-te', '--test_list', type=str,required=True,
                    help="Absolute path of the testing images list with gps information.")
    parser.add_argument('-v', '--vlad_features',type=str, required=True,
                    help="Absolute path of the vlad codebook file")
    parser.add_argument('-k', '--top_k',type=int, default=5,
                    help="Number of nearest neighbors to retrieve")
    parser.add_argument('-d', '--acceptance_distance_thresh',type=float, default=5.0,
                    help="Distnace threshold for localization in meters")
    parser.add_argument('-g', '--gps_type',type=str, default='meters',
                    help="Type of gps coordinates. It can be either meters or global. global means global lat & long")

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    args.similarity_thresh = np.cos(np.deg2rad(75)) #Required threshold of cosine similarity

    #args.input_records = '/uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/data.txt'

    #args.map_videos = [9] #vocabulary was built using videos in this list

    main(args)

