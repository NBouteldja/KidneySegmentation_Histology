# KidneySegmentation_Histology
Python code used to train and evaluate segmentation networks for renal histopathological analysis:  
* training.py --model --setting --epochs --batchSize --lrate --weightDecay  
Script is used to train and possibly evaluate a segmentation network  
* getPredictionForBigPatch.py  
Script is used to compute predictions including all pre- and postprocessing steps within a specified WSI.
# Installation
1. Clone this repo using [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git):<br>
```
git clone https://github.com/NBouteldja/KidneySegmentation_Histology.git
```
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and use conda to create a suitable python environment as prepared in *environment.yml*:<br>
```
conda env create -f ./environment.yml
```
3. Activate installed python environment:
```
source activate python37
```
4. Install [pytorch](https://pytorch.org/) depending on your system:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
# Training
Train a network (Note: Before, you need to specify the path to results folder (variable: *resultsPath*) in *training.py* and the path to your data set folder (variable: *image_dir_base*) in *dataset.py*).
```
python ./KidneySegmentation_Histology/training.py -m custom -s train_val_test -e 500 -b 6 -r 0.001 -w 0.00001
```
Note: *training.py* is specified as follows:
```
training.py --model --setting --epochs --batchSize --lrate --weightDecay 
```
# Application
Use the trained model to segment data following *getPredictionPatch.py*.
<br>
<br>
# Copyright          

    /**************************************************************************
    *                                                                         *
    *   Copyright (C) 2020 by RWTH Aachen University                          *
    *   http://www.rwth-aachen.de                                             *
    *                                                                         *
    *                                                                         *
    *   License:                                                              *
    *                                                                         *
    *   This software is dual-licensed under:                                 *
    *   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)         *
    *   • AGPL (GNU Affero General Public License) open source license        *
    *                                                                         *
    ***************************************************************************/                                                                
