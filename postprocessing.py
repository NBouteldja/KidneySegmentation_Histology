import numpy as np
import torch
import torch.nn as nn
import math
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes

from utils import getChannelSmootingConvLayer


structure = np.zeros((3, 3), dtype=np.int)
structure[1, :] = 1
structure[:, 1] = 1


colors = np.array([    [  0,   0,   0], # Black
                       [255,   0,   0], # Red
                       [  0, 128,   0], # Green
                       [  0,   0, 255], # Blue
                       [  0, 255, 255], # Cyan
                       [255,   0, 255], # Magenta
                       [255, 255,   0], # Yellow
                       [139,  69,  19], # Brown (saddlebrown)
                       [128,   0, 128], # Purple
                       [255, 140,   0], # Orange
                       [255, 255, 255]], dtype=np.uint8) # White


def getRandomTubuliColor():
    while(True):
        candidateColor = np.random.randint(low=0, high=256, size=3, dtype=np.uint8)
        if not ((np.abs((candidateColor-colors[0:7])).sum(1)<50).any()):
            return candidateColor


def extractInstanceChannels(postprocessedPrediction, preprocessedGT, tubuliDilation=True):

    postprocessedPredictionRGB = np.zeros(shape=(preprocessedGT.shape[0], preprocessedGT.shape[1], 3), dtype=np.uint8)
    preprocessedGTrgb = postprocessedPredictionRGB.copy()
    for i in range(2, 7):
        postprocessedPredictionRGB[postprocessedPrediction == i] = colors[i]
        preprocessedGTrgb[preprocessedGT == i] = colors[i]

    labeledTubuli, numberTubuli = label(np.asarray(postprocessedPrediction == 1, np.uint8), structure)
    labeledGlom, _ = label(np.asarray(np.logical_or(postprocessedPrediction == 2, postprocessedPrediction == 3), np.uint8), structure)
    labeledTuft, _ = label(np.asarray(postprocessedPrediction == 3, np.uint8), structure)
    labeledVeins, _ = label(np.asarray(postprocessedPrediction == 4, np.uint8), structure)
    labeledArtery, _ = label(np.asarray(np.logical_or(postprocessedPrediction == 5, postprocessedPrediction == 6), np.uint8), structure)
    labeledArteryLumen, _ = label(np.asarray(postprocessedPrediction == 6, np.uint8), structure)

    for i in range(1, numberTubuli + 1):
        if tubuliDilation:
            tubuliSelection = binary_dilation(labeledTubuli == i)
            labeledTubuli[tubuliSelection] = i
        else:
            tubuliSelection = labeledTubuli == i
        postprocessedPredictionRGB[tubuliSelection] = getRandomTubuliColor()


    labeledTubuliGT, numberTubuliGT = label(np.asarray(preprocessedGT == 1, np.uint8), structure)
    labeledGlomGT, _ = label(np.asarray(np.logical_or(preprocessedGT == 2, preprocessedGT == 3), np.uint8), structure)
    labeledTuftGT, _ = label(np.asarray(preprocessedGT == 3, np.uint8), structure)
    labeledVeinsGT, _ = label(np.asarray(preprocessedGT == 4, np.uint8), structure)
    labeledArteryGT, _ = label(np.asarray(np.logical_or(preprocessedGT == 5, preprocessedGT == 6), np.uint8), structure)
    labeledArteryLumenGT, _ = label(np.asarray(preprocessedGT == 6, np.uint8), structure)

    for i in range(1, numberTubuliGT + 1):
        # if tubuliDilation:
        #     tubuliSelectionGT = binary_dilation(labeledTubuliGT == i)
        # else:
        #     tubuliSelectionGT = labeledTubuliGT == i
        tubuliSelectionGT = labeledTubuliGT == i
        preprocessedGTrgb[tubuliSelectionGT] = getRandomTubuliColor()


    return [labeledTubuli, labeledGlom, labeledTuft, labeledVeins, labeledArtery, labeledArteryLumen], [labeledTubuliGT, labeledGlomGT, labeledTuftGT, labeledVeinsGT, labeledArteryGT, labeledArteryLumenGT], postprocessedPredictionRGB, preprocessedGTrgb



def postprocessPredictionAndGT(prediction, GT, device, predictionsmoothing, holefilling):
    """
    :param prediction: Torch FloatTensor of size 1xCxHxW stored in VRAM/on GPU
    :param GT: HxW ground-truth label map, numpy long tensor
    :return: 1.postprocessed labelmap result (prediction smoothing, removal of small areas, hole filling)
             2.network output prediction (w/o postprocessing)
    """
    ################# PREDICTION SMOOTHING ################
    if predictionsmoothing:
        smoothingKernel = getChannelSmootingConvLayer(8).to(device)
        prediction = smoothingKernel(prediction)

    labelMap = torch.argmax(prediction, dim=1).squeeze(0).to("cpu").numpy() # Label 0/1/2/3/4/5/6/7: Background/tubuli/glom_full/glom_tuft/veins/artery_full/artery_lumen/border

    netOutputPrediction = labelMap.copy()

    ################# REMOVING TOO SMALL CONNECTED REGIONS ################
    # Tuft
    labeledTubuli, numberTubuli = label(np.asarray(labelMap == 3, np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, numberTubuli + 1):
        tubuliSelection = (labeledTubuli == i)
        if tubuliSelection.sum() < 500:  # remove too small noisy regions
            labelMap[tubuliSelection] = 2

    # Glomeruli
    labeledTubuli, numberTubuli = label(np.asarray(np.logical_or(labelMap == 3, labelMap==2), np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, numberTubuli + 1):
        tubuliSelection = (labeledTubuli == i)
        if tubuliSelection.sum() < 1500:  # remove too small noisy regions
            labelMap[tubuliSelection] = 0

    # Artery lumen
    labeledTubuli, numberTubuli = label(np.asarray(labelMap == 6, np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, numberTubuli + 1):
        tubuliSelection = (labeledTubuli == i)
        if tubuliSelection.sum() < 20:  # remove too small noisy regions
            labelMap[tubuliSelection] = 5

    # Full artery
    labeledTubuli, numberTubuli = label(np.asarray(np.logical_or(labelMap == 5, labelMap==6), np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, numberTubuli + 1):
        tubuliSelection = (labeledTubuli == i)
        if tubuliSelection.sum() < 400:  # remove too small noisy regions
            labelMap[tubuliSelection] = 0

    # Veins
    labeledTubuli, numberTubuli = label(np.asarray(labelMap == 4, np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, numberTubuli + 1):
        tubuliSelection = (labeledTubuli == i)
        if tubuliSelection.sum() < 3000:  # remove too small noisy regions
            labelMap[tubuliSelection] = 0

    # Tubuli
    labeledTubuli, numberTubuli = label(np.asarray(labelMap == 1, np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, numberTubuli + 1):
        tubuliSelection = (labeledTubuli == i)
        if tubuliSelection.sum() < 400:  # remove too small noisy regions
            labelMap[tubuliSelection] = 0

    ################# HOLE FILLING ################
    if holefilling:
        labelMap[binary_fill_holes(labelMap==1)] = 1 #tubuli
        labelMap[binary_fill_holes(labelMap==4)] = 4 #veins
        tempTuftMask = binary_fill_holes(labelMap==3) #tuft
        labelMap[binary_fill_holes(np.logical_or(labelMap==3, labelMap==2))] = 2 #glom
        labelMap[tempTuftMask] = 3 #tuft
        tempArteryLumenMask = binary_fill_holes(labelMap == 6)  #artery_lumen
        labelMap[binary_fill_holes(np.logical_or(labelMap == 5, labelMap == 6))] = 5  #full_artery
        labelMap[tempArteryLumenMask] = 6  #artery_lumen


    return labelMap, netOutputPrediction, GT





