# KidneySegmentation_Histology
This repository represents a python framework to train, evaluate and apply segmentation networks for renal histological analysis. In particular, we trained a neural network based on the [U-net architecture](https://arxiv.org/pdf/1505.04597.pdf) to segment several renal structures including tubulus ![#ff0000](https://via.placeholder.com/15/ff0000/000000?text=+), glomerulus ![#00ff00](https://via.placeholder.com/15/00ff00/000000?text=+), glomerular tuft ![#0000ff](https://via.placeholder.com/15/0000ff/000000?text=+), vein (including renal pelvis) ![#00ffff](https://via.placeholder.com/15/00ffff/000000?text=+), artery ![#ff00ff](https://via.placeholder.com/15/ff00ff/000000?text=+), and arterial lumen ![#ffff00](https://via.placeholder.com/15/ffff00/000000?text=+) from histopathology data. In our experiments, we utilized murine data sampled from several common disease models as well as healthy data from other species including rat, bear, pig, marmoset and human (MCD) ultimately proving a (murine) multi-disease, multi-species and multi-class segmentation network for renal quantification.

# Installation
1. Clone this repo using [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git):<br>
```
git clone https://github.com/NBouteldja/KidneySegmentation_Histology.git
```
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and use conda to create a suitable python environment as prepared in *environment.yml* that lists all library dependencies:<br>
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
Train a network, e.g. using the following command:
```
python ./KidneySegmentation_Histology/training.py -m custom -s train_val_test -e 500 -b 6 -r 0.001 -w 0.00001
```
Note:<br>
- Before, you need to specify the path to results folder (variable: *resultsPath*) in *training.py* and the path to your data set folder (variable: *image_dir_base*) in *dataset.py*
- *training.py* is parameterized as follows:
```
training.py --model --setting --epochs --batchSize --lrate --weightDecay 
```
# Application
Use *getPredictionForBigPatch.py* to apply the trained network for histopathological renal structure segmentation to data of your choice.
```
python ./KidneySegmentation_Histology/getPredictionForBigPatch.py
```
Note: Before running the script, you need to specify the path to the WSI (variable: *WSIpath*), the network path (variable: *modelpath*), and the path to a results folder (variable: *resultspath*).<br>
In particular, the script will segment a specified patch from the given WSI using the network. Determine the position of the patch of interest by providing the raw coodinates (e.g. coordinates shown in QuPath) of its upper left corner (variable: *patchCenterCoordinatesRaw*) and determine its size by modifying *patchGridCellTimes*. The latter variable specifies how many 516x516 patches are segmented row-wise as well as column-wise.<br>
<br>
You can also apply the trained network to our provided exemplary image patches contained in the folder *exemplaryData*. These patches show various pathologies associated with different murine disease models, and are listed below including our ground-truth annotation:
<br>
| Healthy | Annotation |
|:--:|:--:|
| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/Healthy.png?raw=true" width="400">| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/Healthy-labels.png?raw=true" width="324"> |
| UUO | Annotation |
| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/UUO.png?raw=true" width="400">| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/UUO-labels.png?raw=true" width="324"> |
| Adenine | Annotation |
| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/Adenine.png?raw=true" width="400">| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/Adenine-labels.png?raw=true" width="324"> |
| Alport | Annotation |
| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/Alport.png?raw=true" width="400">| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/Alport-labels.png?raw=true" width="324"> |
| IRI | Annotation |
| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/IRI.png?raw=true" width="400">| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/IRI-labels.png?raw=true" width="324"> |
| NTN | Annotation |
| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/NTN.png?raw=true" width="400">| <img src="https://github.com/NBouteldja/KidneySegmentation_Histology/blob/master/exemplaryData/NTN-labels.png?raw=true" width="324"> |

# Contact
Peter Boor, MD, PhD<br>
Institute of Pathology<br>
RWTH Aachen University Hospital<br>
Pauwelsstrasse 30<br>
52074 Aachen, Germany<br>
Phone:	+49 241 80 85227<br>
Fax:		+49 241 80 82446<br>
E-mail: 	pboor@ukaachen.de<br>
<br>

#
    /**************************************************************************
    *                                                                         *
    *   Copyright (C) 2020 by RWTH Aachen University                          *
    *   http://www.rwth-aachen.de                                             *
    *                                                                         *
    *   License:                                                              *
    *                                                                         *
    *   This software is dual-licensed under:                                 *
    *   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)         *
    *   • AGPL (GNU Affero General Public License) open source license        *
    *                                                                         *
    ***************************************************************************/                                                                
