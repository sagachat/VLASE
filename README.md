# VLASE  
VLASE: Vehicle Localization by Aggregating Semantic Edges   
  
## Pre-requisites:  
1. [Casenet](https://arxiv.org/pdf/1705.09759.pdf)  
2. opencv-2.4.13  
3. Python libraries as listed in the requirements.txt  
  
## Installation:  
  
  
## Casenet Usage:  

### Casenet Class Ids: 

Casenet is pre-trained on [cityscapes](https://www.cityscapes-dataset.com/) and following classes (with associated class ids) were used:   
  
#### Static :  
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
  
#### Dynamic:  
11 -> 'person'  
12 -> 'rider'  
13 -> 'car'  
14 -> 'truck'  
15 -> 'bus'  
16 -> 'train'  
17 -> 'motorcycle'  
18 -> 'bicycle'  
  
  
## Data Preparation:  
  
  
  
  
  
  
  
  
  

  
## Types of experiments:  

### Sift
- Standard  
- Sift with changing the value of alpha (Assign different weights to sift features and xy coordinate features)  
- Sift with changing the number of clusters  
  
### Casenet
- Standard  
- Casenet with changing the value of alpha (Assign different weights to casenet features and xy coordinate features)  
- Casenet with changing the number of clusters  
- Casenet with removing a subset of features (All Static, Combination of some static)  
  
  
## Execution Instructions:  
  
### Build Vocab:  
  
#### Using Casenet   
  
##### Standard
python build_vlad_vocabulary.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09_GPS.txt -r /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09 -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/AM09/resized -f casenet -c 64 -s 5  
  
##### Remove features -   
python build_vlad_vocabulary.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09_GPS.txt -r /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09 -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/AM09/resized -f casenet -c 64 -s 5  -crf 11,12,13  
  
##### Change weights of casenet features and XY using alpha
python build_vlad_vocabulary.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09_GPS.txt -r /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09 -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/AM09/resized -f casenet -c 64 -s 5  -a 0.1  
  
#### Using Sift   
  
##### Standard  
python build_vlad_vocabulary.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09_GPS.txt -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/AM09/resized -s 5  
  
##### Change weights of SIFT features and XY using alpha
python build_vlad_vocabulary.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/AM09_GPS.txt -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/AM09/resized -s 5  -a 0.1  
  
  
### Create VLAD descriptors:  
  
While creating VLAD descriptors, the script's input list and input directories must have both training and testing data.  
  
#### Using Casenet   
  
##### Standard
python compute_vlad_descriptor.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/data.txt -r /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/all -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/all -f casenet -c 64 -s 5  
  
##### Remove features -   
python compute_vlad_descriptor.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/data.txt -r /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/all -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/all -f casenet -c 64 -s 5 -crf 11,12,13  
  
##### Change weights of casenet features and XY using alpha
python compute_vlad_descriptor.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/data.txt -r /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/all -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/all -f casenet -c 64 -s 5  -a 0.1  
  
  
  
#### Using Sift   
  
##### Standard  
python compute_vlad_descriptor.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/data.txt -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/all -s 5  
  
##### Change weights of SIFT features and XY using alpha
python compute_vlad_descriptor.py -i /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST_CASENET/data.txt -src /uusoc/exports/scratch/xiny/cvpr18-localization_dataset/KAIST/KAIST/all -s 5  -a 0.1  
  

### Localize:  
  
  
  
  
  
## Demo Video:  
  
  
  
  
  
  
  
  
  
  