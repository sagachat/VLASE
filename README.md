# VLASE: Vehicle Localization by Aggregating Semantic Edges   

VLASE is a framework to use semantic edge features from images to achieve on-road localization. Semantic edge features denote edge contours that separate pairs of distinct objects such as building-sky, roadsidewalk, and building-ground. While prior work has shown promising results by utilizing the boundary between prominent
classes such as sky and building using skylines, we generalize this approach to consider semantic edge features that arise from 19 different classes. Our localization algorithm is simple, yet very powerful. We extract semantic edge features using a recently introduced CASENet architecture and utilize VLAD framework to perform image retrieval. Our experiments show that we achieve improvement over some of the state-of-the-art localization algorithms such as SIFT-VLAD and its deep variant NetVLAD. We use ablation study to study the importance of different semantic classes, and show that our unified approach achieves better performance compared to individual prominent features such as skylines.

Here is the demo video for VLASE - https://youtu.be/IKZXZmmdtiA
  
## Pre-requisites:  
1. [Casenet](http://www.merl.com/research/license#CASENet)  
2. opencv-2.4.13  - Required for SIFT features and image processing
3. Python libraries as listed in the [requirements.txt](https://github.com/sagachat/VLASE/blob/master/requirements.txt)
  
## Installation:  
1. Install Python 2.7 and the libraries listed in the [requirements.txt](https://github.com/sagachat/VLASE/blob/master/requirements.txt)
2. Install OpenCV 2.4.13. We prefer to build it from source.
2. Install CASENet as per the instructions in it's README file. CASENet includes it's own Caffe version.

## Casenet: 

Casenet is pre-trained on [cityscapes](https://www.cityscapes-dataset.com/) and following classes (with associated class ids) were used:   
  
### Static classes:  
0 -> 'road'  
1 -> 'sidewalk'  
2 -> 'building'  
3 -> 'wall'  
4 -> 'fence'  
5 -> 'pole'  
6 -> 'traffic light'  
7 -> 'traffic sign'  
8 -> 'vegetation'  
9 -> 'terrain'  
10 -> 'sky'  
  
### Dynamic classes:  
11 -> 'person'  
12 -> 'rider'  
13 -> 'car'  
14 -> 'truck'  
15 -> 'bus'  
16 -> 'train'  
17 -> 'motorcycle'  
18 -> 'bicycle'  
 
## Data Preparation:  
1. Extract all the frames from the source dashcam video into a directory (We'll call it src). VLASE uses original frames for SIFT baselines.
2. If you want to you VLASE with CASENet features then extract the CASENet features for the original frames into a directory (We'll call it result_root). result_root will have a directory structure like following -
   ```
    |-- class_0
    |-- class_1
    |-- class_10
    |-- class_11
    |-- class_12
    |-- class_13
    |-- class_14
    |-- class_15
    |-- class_16
    |-- class_17
    |-- class_18
    |-- class_2
    |-- class_3
    |-- class_4
    |-- class_5
    |-- class_6
    |-- class_7
    |-- class_8
    `-- class_9
   ```
   These 19 directories contain features of each frame for 19 semantic classes. If there are total N frames then each of these directories will have N files.
3. VLASE trains and tests on different sets of frames. So split your data into training and testing images.
4. You must have a training and a testing data file with the GPS information of the frames. These files must have the data in following space separated format:
   ```
   Frame_Name X Y
   AM09_000000.png 36.37261637 127.3641256
   AM09_000001.png 36.37261636 127.3641256
   ```
   Refer to the directory [sample_data](https://github.com/sagachat/VLASE/blob/master/sample_data/) for sample training and testing files.

5. X and Y are the gps coordinates of the frames. These can be global latitude and longitude or they can be in meters (relative to the starting frame).
  
## Types of experiments:  

### Sift
- Standard: All features equally weighted
- Sift with changing the value of alpha (Assign different weights to sift features and xy coordinate features)  
  
### Casenet
- Standard: All features equally weighted
- Casenet with changing the value of alpha (Assign different weights to casenet features and xy coordinate features)  
- Casenet with removing a subset of features (All Static, Combination of some static)  
  
  
## Execution Instructions:  
  
### Build Vocab:  scripts/build_vlad_vocabulary.py
First of all, VLASE builds vocabulary using the training frames. Following are the input arguments for this script -

##### Required
   
   ```
    '-i', '--input_list', type=str,required=True,help=Absolute path of the training images list with gps information
    '-src', '--src', type=str,required=True,help=Directory containing original source training images
   ```

##### Casenet specific arguments
   ```
    '-r', '--result_root', type=str,default=None,help=Directory containing casenet features of training images
    '-t', '--thresh', type=float, default='0.5',help=Probability threshold value to use casenet features
    '-rc', '--result_classes', type=int, default=19,help=Number of casenet classes
    '-crf', '--removed_class',type=str, default=None,help=Comma seperated list of casenet class ids that should be removed during the experiment
   ```
##### Optional
   ```
    '-f','--feature_type', type=str, default='sift',help=Feature type used for localization - casenet or sift. By default, the script uses SIFT features.

    '-c', '--vlad_feature_dimension', type=int, default='32',help=Number of clusters for VLAD. In our experiments, SIFT performed best with cluster=32 and Casenet was best with cluster=64.
 
    '-a', '--alpha', type=float, default='0.5',help=Alpha gives weight to the features (Casenet/SIFT vs XY).  XY will have weight = (1-alpha).alpha=0.5 means equal weight to both. In our experiments, alpha=0.1 gave the best results.

    '-s', '--skip_count', type=int, default='0',help=Number of frames to skip while training and testing VLAD. To reduce the execution time, we skip video frames. It did not make much difference in the performance as we skip according to the frame rate of videos. For e.g. KAIST dataset has frame rate = 10, so we skip 5 frames while creating vocabulary as it does not cause much loss of information.
    ```
Examples -
    
    Using Casenet with Removing features -   
    python build_vlad_vocabulary.py -i /home/sagar/KAIST/train_GPS.txt -r /home/sagar/KAIST/KAIST_CASENET/train -src /home/sagar/KAIST/src -f casenet -c 64 -s 5  -crf 11,12,13  
  
    Using Sift Standard -
    python build_vlad_vocabulary.py -i /home/sagar/KAIST/train_GPS.txt -src /home/sagar/KAIST/src -s 5  
  
   ``` 
  
  
### Create VLAD descriptors:  
After building vocab, VLASE creates the VLAD descriptors for training and testing frames. Following are the input arguments for this script -

##### Required
   
   ```
    '-tr', '--train_list', type=str,required=True,help=Absolute path of the training images list with gps information. This is the same list used for building vocabulary.
    '-te', '--test_list', type=str,required=True,help=Absolute path of the testing images list with gps information.
    '-src', '--src', type=str,required=True,help=Directory containing original source images, this directory must have both training and testing data.
   ```

##### Casenet specific arguments
   ```
    '-r', '--result_root', type=str,default=None,help=Directory containing casenet features, this directory must have both training and testing data.
    '-t', '--thresh', type=float, default='0.5',help=Probability threshold value to use casenet features
    '-rc', '--result_classes', type=int, default=19,help=Number of casenet classes
    '-crf', '--removed_class',type=str, default=None,help=Comma seperated list of casenet class ids that should be removed during the experiment
   ```
##### Optional
   ```
    '-f','--feature_type', type=str, default='sift',help=Feature type used for localization - casenet or sift. By default, the script uses SIFT features.

    '-c', '--vlad_feature_dimension', type=int, default='32',help=Number of clusters for VLAD. In our experiments, SIFT performed best with cluster=32 and Casenet was best with cluster=64.
 
    '-a', '--alpha', type=float, default='0.5',help=Alpha gives weight to the features (Casenet/SIFT vs XY).  XY will have weight = (1-alpha).alpha=0.5 means equal weight to both. In our experiments, alpha=0.1 gave the best results.

    '-s', '--skip_count', type=int, default='0',help=Number of frames to skip while training and testing VLAD. To reduce the execution time, we skip video frames. It did not make much difference in the performance as we skip according to the frame rate of videos. For e.g. KAIST dataset has frame rate = 10, so we skip 5 frames while creating vocabulary as it does not cause much loss of information.
    ```
Examples -
    
    Using Casenet with Removing features -   
    python compute_vlad_descriptor.py -tr /home/sagar/KAIST/train_GPS.txt -te /home/sagar/KAIST/test_GPS.txt -r /home/sagar/KAIST/KAIST_CASENET/all -src /home/sagar/KAIST/all -f casenet -c 64 -s 5 -crf 11,12,13  
  
    Using Sift Standard -
    python compute_vlad_descriptor.py -tr /home/sagar/KAIST/train_GPS.txt -te /home/sagar/KAIST/test_GPS.txt -src /home/sagar/KAIST/all -s 5  
  
   ``` 
  

### Localize:  
  
#### Global gps lat & long
python vlad_place_recognition.py -tr /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09_GPS.txt -te /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM05_GPS.txt -v vocabulary_f=casenet_k=64_t=0.5_alpha=0.5.vlad.npz -g global

#### gps in meters
python vlad_place_recognition.py -tr /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09_GPS.txt -te /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM05_GPS.txt -v vocabulary_f=casenet_k=64_t=0.5_alpha=0.5.vlad.npz

#### For a specific distance threshold
python vlad_place_recognition.py -tr /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09_GPS.txt -te /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM05_GPS.txt -v vocabulary_f=casenet_k=64_t=0.5_alpha=0.5.vlad.npz -g global -d 10
  
    
  
  
  
  
  
  
  
  
