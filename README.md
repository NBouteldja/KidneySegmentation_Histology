# KidneySegmentation_Histology
Python code used to train and evaluate segmentation networks for renal histopathological analysis.<br>
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
Use *getPredictionForBigPatch.py* to apply the trained network for histopathological renal structure segmentation on data of your choice:
```
python ./KidneySegmentation_Histology/getPredictionForBigPatch.py
```
You can also use our provided exemplary image data from the folder *exemplaryData* showing various specific pathologies associated with our different murine disease models. (Note: Before running the script, you need to specify the path to the WSI (variable: *WSIpath*), the model path (variable: *modelpath*), and the path to a results folder (variable: *resultspath*).)
<br>
<br>
#           

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
