# KidneySegmentation_Histology
Python code used to train and evaluate segmentation networks for renal histopathological analysis:  
* training.py --model --setting --epochs --batchSize --lrate --weightDecay  
Script is used to train and possibly evaluate a segmentation network  
* getPredictionForBigPatch.py  
Script is used to compute predictions including all pre- and postprocessing steps within a specified WSI.
# Installation
1. Clone this repo with git:<br>
```
git clone https://github.com/NBouteldja/KidneySegmentation_Histology.git
```
2. Install miniconda and use conda to create a suitable python environment:<br>
```
conda env create -f ./environment.yml
```
<br>
<br>                 

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
