import numpy as np
import torch
from subprocess import check_output
import os
import psutil
import math
from skimage.util import view_as_windows
from itertools import product
from typing import Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes


colors = torch.tensor([[  0,   0,   0], # Black
                       [255,   0,   0], # Red
                       [  0, 255,   0], # Green
                       [  0,   0, 255], # Blue
                       [  0, 255, 255], # Cyan
                       [255,   0, 255], # Magenta
                       [255, 255,   0], # Yellow
                       [139,  69,  19], # Brown (saddlebrown)
                       [128,   0, 128], # Purple
                       [255, 140,   0], # Orange
                       [255, 255, 255]], dtype=torch.uint8) # White


def generate_ball(radius):
    structure = np.zeros((3, 3), dtype=np.int)
    structure[1, :] = 1
    structure[:, 1] = 1

    ball = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    ball[radius, radius] = 1
    for i in range(radius):
        ball = binary_dilation(ball, structure=structure)
    return np.asarray(ball, dtype=np.int)
  
  
def convert_labelmap_to_rgb(labelmap):
    """
    Method used to generate rgb label maps for tensorboard visualization
    :param labelmap: HxW label map tensor containing values from 0 to n_classes
    :return: 3xHxW RGB label map containing colors in the following order: Black (background), Red, Green, Blue, Cyan, Magenta, Yellow, Brown, Orange, Purple
    """
    n_classes = labelmap.max()

    result = torch.zeros(size=(labelmap.size()[0], labelmap.size()[1], 3), dtype=torch.uint8)
    for i in range(1, n_classes+1):
        result[labelmap == i] = colors[i]

    return result.permute(2, 0, 1)

def convert_labelmap_to_rgb_with_instance_first_class(labelmap, structure):
    """
    Method used to generate rgb label maps for tensorboard visualization
    :param labelmap: HxW label map tensor containing values from 0 to n_classes
    :return: 3xHxW RGB label map containing colors in the following order: Black (background), Red, Green, Blue, Cyan, Magenta, Yellow, Brown, Orange, Purple
    """
    n_classes = labelmap.max()

    result = np.zeros(shape=(labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for i in range(2, n_classes+1):
        result[labelmap == i] = colors[i].numpy()

    structure = np.ones((3, 3), dtype=np.int)

    labeledTubuli, numberTubuli = label(np.asarray(labelmap == 1, np.uint8), structure)  # datatype of 'labeledTubuli': int32
    for i in range(1, numberTubuli + 1):
        result[binary_dilation(binary_dilation(binary_dilation(labeledTubuli == i)))] = np.random.randint(low=0, high=256, size=3, dtype=np.uint8)  # assign random colors to tubuli

    return result

def convert_labelmap_to_rgb_except_first_class(labelmap):
    """
    Method used to generate rgb label maps for tensorboard visualization
    :param labelmap: HxW label map tensor containing values from 0 to n_classes
    :return: 3xHxW RGB label map containing colors in the following order: Black (background), Red, Green, Blue, Cyan, Magenta, Yellow, Brown, Orange, Purple
    """
    n_classes = labelmap.max()

    result = torch.zeros(size=(labelmap.size()[0], labelmap.size()[1], 3), dtype=torch.uint8)
    for i in range(2, n_classes+1):
        result[labelmap == i] = colors[i]

    return result.permute(2, 0, 1)


def getColorMapForLabelMap():
    return ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'brown', 'orange', 'purple', 'white']

def saveFigureResults(img, outputPrediction, postprocessedPrediction, finalPredictionRGB, GT, preprocessedGT, preprocessedGTrgb, fullResultPath, alpha=0.4):
    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert outputPrediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    # avoid brown color (border visualization) in output for final GT and prediction
    postprocessedPrediction[postprocessedPrediction==7] = 0
    preprocessedGT[preprocessedGT==7] = 0

    # also dilate tubuli here
    postprocessedPrediction[binary_dilation(postprocessedPrediction==1)] = 1

    predictionMask = np.ma.masked_where(postprocessedPrediction == 0, postprocessedPrediction)

    plt.figure(figsize=(16, 8.1))
    plt.subplot(241)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(outputPrediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(postprocessedPrediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(finalPredictionRGB)
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(img[(img.shape[0]-outputPrediction.shape[0])//2:(img.shape[0]-outputPrediction.shape[0])//2+outputPrediction.shape[0],(img.shape[1]-outputPrediction.shape[1])//2:(img.shape[1]-outputPrediction.shape[1])//2+outputPrediction.shape[1],:])
    plt.imshow(predictionMask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha = alpha)
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(GT, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(preprocessedGT, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(preprocessedGTrgb)
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(fullResultPath)
    plt.close()

def savePredictionResults(predictionWithoutTubuliDilation, fullResultPath, figSize):
    prediction = predictionWithoutTubuliDilation.copy()
    prediction[binary_dilation(binary_dilation(binary_dilation(binary_dilation(prediction == 1))))] = 1

    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert prediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(prediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.savefig(fullResultPath)
    plt.close()

def savePredictionResultsWithoutDilation(prediction, fullResultPath, figSize):
    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert prediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(prediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.savefig(fullResultPath)
    plt.close()

def savePredictionOverlayResults(img, predictionWithoutTubuliDilation, fullResultPath, figSize, alpha=0.4):
    prediction = predictionWithoutTubuliDilation.copy()
    prediction[binary_dilation(binary_dilation(binary_dilation(prediction == 1)))] = 1
    predictionMask = np.ma.masked_where(prediction == 0, prediction)

    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert prediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    ax.imshow(predictionMask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha = alpha)
    plt.savefig(fullResultPath)
    plt.close()

def saveOverlayResults(img, seg, fullResultPath, figHeight, alpha=0.4):
    segMask = np.ma.masked_where(seg == 0, seg)

    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert seg.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    fig = plt.figure(figsize=(figHeight*seg.shape[1]/seg.shape[0], figHeight))
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    ax.imshow(segMask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha = alpha)
    plt.savefig(fullResultPath)
    plt.close()

def saveRGBPredictionOverlayResults(img, prediction, fullResultPath, figSize, alpha=0.4):
    predictionMask = prediction.sum(2)==0
    predictionCopy = prediction.copy()
    predictionCopy[predictionMask] = img[predictionMask]
    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.asarray(np.round(predictionCopy*alpha+(1-alpha)*img), np.uint8))
    plt.savefig(fullResultPath)
    plt.close()

def saveImage(img, fullResultPath, figSize):
    fig = plt.figure(figsize=figSize)
    ax = plt.Axes(fig, [0., 0., 1., 1., ])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    plt.savefig(fullResultPath)
    plt.close()



def getCrossValSplits(dataIDX, amountFolds, foldNo, setting):
    """
    Cross-Validation-Split of indices according to fold number and setting
    Usage:
        dataIDX = np.arange(dataset.__len__())
        # np.random.shuffle(dataIDX)
        for i in range(amountFolds):
            train_idx, val_idx, test_idx = getCrossFoldSplits(dataIDX=dataIDX, amountFolds=amountFolds, foldNo=i+1, setting=setting)
    :param dataIDX: All data indices stored in numpy array
    :param amountFolds: Total amount of folds
    :param foldNo: Fold number, # CARE: Fold numbers start with 1 and go up to amountFolds ! #
    :param setting: Train / Train+Test / Train+Val / Train+Test+Val
    :return: tuple consisting of 3 numpy arrays (trainIDX, valIDX, testIDX) containing indices according to split
    """
    assert (setting in ['train_val_test', 'train_test', 'train_val', 'train']), 'Given setting >'+setting+'< is incorrect!'

    num_total_data = dataIDX.__len__()

    if setting == 'train':
        return dataIDX, None, None

    elif setting == 'train_val':
        valIDX = dataIDX[num_total_data * (foldNo - 1) // amountFolds: num_total_data * foldNo // amountFolds]
        trainIDX = np.setxor1d(dataIDX, valIDX)
        return trainIDX, valIDX, None

    elif setting == 'train_test':
        testIDX = dataIDX[num_total_data * (foldNo - 1) // amountFolds: num_total_data * foldNo // amountFolds]
        trainIDX = np.setxor1d(dataIDX, testIDX)
        return trainIDX, None, testIDX

    elif setting == 'train_val_test':
        testIDX = dataIDX[num_total_data * (foldNo - 1) // amountFolds: num_total_data * foldNo // amountFolds]
        if foldNo != amountFolds:
            valIDX = dataIDX[num_total_data * foldNo // amountFolds: num_total_data * (foldNo+1) // amountFolds]
        else:
            valIDX = dataIDX[0 : num_total_data // amountFolds]
        trainIDX = np.setxor1d(np.setxor1d(dataIDX, testIDX), valIDX)
        return trainIDX, valIDX, testIDX

    else:
        raise ValueError('Given setting >'+str(setting)+'< is invalid!')


def parse_nvidia_smi(unit=0):
    result = check_output(["nvidia-smi", "-i", str(unit),]).decode('utf-8').split('\n')
    return 'Current GPU usage: ' + result[0] + '\r\n' + result[5] + '\r\n' + result[8]


def parse_RAM_info():
    return 'Current RAM usage: '+str(round(psutil.Process(os.getpid()).memory_info().rss / 1E6, 2))+' MB'


def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def getOneHotEncoding(imgBatch, labelBatch):
    """
    :param imgBatch: image minibatch (FloatTensor) to extract shape and device info for output
    :param labelBatch: label minibatch (LongTensor) to be converted to one-hot encoding
    :return: One-hot encoded label minibatch with equal size as imgBatch and stored on same device
    """
    if imgBatch.size()[1] != 1: # Multi-label segmentation otherwise binary segmentation
        labelBatch = labelBatch.unsqueeze(1)
        onehotEncoding = torch.zeros_like(imgBatch)
        onehotEncoding.scatter_(1, labelBatch, 1)
        return onehotEncoding
    return labelBatch


def getWeightsForCEloss(dataset, train_idx, areLabelsOnehotEncoded, device, logger):
    # Choice 1) Manually set custom weights
    weights = torch.tensor([1,2,4,6,2,3], dtype=torch.float32, device=device)
    weights = weights / weights.sum()

    # Choice 2) Compute weights as "np.mean(histogram) / histogram"
    dataloader = DataLoader(dataset=dataset, batch_size=6, sampler=SubsetRandomSampler(train_idx), num_workers=6)

    if areLabelsOnehotEncoded:
        histograms = 0
        for batch in dataloader:
            imgBatch, segBatch = batch
            amountLabels = segBatch.size()[1]
            if amountLabels == 1: # binary segmentation
                histograms = histograms + torch.tensor([(segBatch==0).sum(),(segBatch==1).sum()])
            else: # multi-label segmentation
                if imgBatch.dim() == 4: #2D data
                    histograms = histograms + segBatch.sum(3).sum(2).sum(0)
                else: #3D data
                    histograms = histograms + segBatch.sum(4).sum(3).sum(2).sum(0)

        histograms = histograms.numpy()
    else:
        histograms = np.array([0])
        for batch in dataloader:
            _, segBatch = batch

            segHistogram = np.histogram(segBatch.numpy(), segBatch.numpy().max()+1)[0]

            if len(histograms) >= len(segHistogram): #(segHistogram could have different size than histograms)
                histograms[:len(segHistogram)] += segHistogram
            else:
                segHistogram[:len(histograms)] += histograms
                histograms = segHistogram

    weights = np.mean(histograms) / histograms
    weights = torch.from_numpy(weights).float().to(device)

    logger.info('=> Weights for CE-loss: '+str(weights))

    return weights



def getMeanDiceScores(diceScores, logger):
    """
    Compute mean label dice scores of numpy dice score array (2d) (and its mean)
    :return: mean label dice scores with '-1' representing totally missing label (meanLabelDiceScores), mean overall dice score (meanOverallDice)
    """
    meanLabelDiceScores = np.ma.masked_where(diceScores == -1, diceScores).mean(0).data
    label_GT_occurrences = (diceScores != -1).sum(0)
    if (label_GT_occurrences == 0).any():
        logger.info('[# WARNING #] Label(s): ' + str(np.argwhere(label_GT_occurrences == 0).flatten() + 1) + ' not present at all in current dataset split!')
        meanLabelDiceScores[label_GT_occurrences == 0] = -1
    meanOverallDice = meanLabelDiceScores[meanLabelDiceScores != -1].mean()

    return meanLabelDiceScores, meanOverallDice


def getDiceScores(prediction, segBatch):
    """
    Compute mean dice scores of predicted foreground labels.
    NOTE: Dice scores of missing gt labels will be excluded and are thus represented by -1 value entries in returned dice score matrix!
    NOTE: Method changes prediction to 0/1 values in the binary case!
    :param prediction: BxCxHxW (if 2D) or BxCxHxWxD (if 3D) FloatTensor (care: prediction has not undergone any final activation!) (note: C=1 for binary segmentation task)
    :param segBatch: BxCxHxW (if 2D) or BxCxHxWxD (if 3D) FloatTensor (Onehot-Encoding) or Bx1xHxW (if 2D) or Bx1xHxWxD (if 3D) LongTensor
    :return: Numpy array containing BxC-1 (background excluded) dice scores
    """
    batchSize, amountClasses = prediction.size()[0], prediction.size()[1]

    if amountClasses == 1: # binary segmentation task => simulate sigmoid to get label results
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = 0
        prediction = prediction.squeeze(1)
        segBatch = segBatch.squeeze(1)
        amountClasses += 1
    else: # multi-label segmentation task
        prediction = prediction.argmax(1) # LongTensor without C-channel
        if segBatch.dtype == torch.float32:  # segBatch is onehot-encoded
            segBatch = segBatch.argmax(1)
        else:
            segBatch = segBatch.squeeze(1)

    prediction = prediction.view(batchSize, -1)
    segBatch = segBatch.view(batchSize, -1)

    labelDiceScores = np.zeros((batchSize, amountClasses-1), dtype=np.float32) - 1 #ignore background class!
    for b in range(batchSize):
        currPred = prediction[b,:]
        currGT = segBatch[b,:]

        for c in range(1,amountClasses):
            classPred = (currPred == c).float()
            classGT = (currGT == c).float()

            if classGT.sum() != 0: # only evaluate label prediction when is also present in ground-truth
                labelDiceScores[b, c-1] = ((2. * (classPred * classGT).sum()) / (classGT.sum() + classPred.sum())).item()

    return labelDiceScores


def printResultsForDiseaseModel(evaluatorID, allClassEvaluators, applyTestTimeAugmentation, logger, saveResults, resultsPath, diseaseModels):
    logger.info('########## NOW: Detection (average precision) and segmentation accuracies (object-level dice): ##########')
    precisionsTub, avg_precisionTub, avg_dice_scoreTub, std_dice_scoreTub, min_dice_scoreTub, max_dice_scoreTub = allClassEvaluators[evaluatorID][0].score()  # tubuliresults
    precisionsGlom, avg_precisionGlom, avg_dice_scoreGlom, std_dice_scoreGlom, min_dice_scoreGlom, max_dice_scoreGlom = allClassEvaluators[evaluatorID][1].score()  # tubuliresults
    precisionsTuft, avg_precisionTuft, avg_dice_scoreTuft, std_dice_scoreTuft, min_dice_scoreTuft, max_dice_scoreTuft = allClassEvaluators[evaluatorID][2].score()  # tubuliresults
    precisionsVeins, avg_precisionVeins, avg_dice_scoreVeins, std_dice_scoreVeins, min_dice_scoreVeins, max_dice_scoreVeins = allClassEvaluators[evaluatorID][3].score()  # tubuliresults
    precisionsArtery, avg_precisionArtery, avg_dice_scoreArtery, std_dice_scoreArtery, min_dice_scoreArtery, max_dice_scoreArtery = allClassEvaluators[evaluatorID][4].score()  # tubuliresults
    precisionsLumen, avg_precisionLumen, avg_dice_scoreLumen, std_dice_scoreLumen, min_dice_scoreLumen, max_dice_scoreLumen = allClassEvaluators[evaluatorID][5].score()  # tubuliresults
    logger.info('DETECTION RESULTS MEASURED BY AVERAGE PRECISION:')
    logger.info('0.5    0.55    0.6    0.65    0.7    0.75    0.8    0.85    0.9 <- Thresholds')
    logger.info(str(np.round(precisionsTub, 4)) + ', Mean: ' + str(np.round(avg_precisionTub, 4)) + '  <-- Tubuli')
    logger.info(str(np.round(precisionsGlom, 4)) + ', Mean: ' + str(np.round(avg_precisionGlom, 4)) + '  <-- Glomeruli (incl. tuft)')
    logger.info(str(np.round(precisionsTuft, 4)) + ', Mean: ' + str(np.round(avg_precisionTuft, 4)) + '  <-- Tuft')
    logger.info(str(np.round(precisionsVeins, 4)) + ', Mean: ' + str(np.round(avg_precisionVeins, 4)) + '  <-- Veins')
    logger.info(str(np.round(precisionsArtery, 4)) + ', Mean: ' + str(np.round(avg_precisionArtery, 4)) + '  <-- Artery (incl. lumen)')
    logger.info(str(np.round(precisionsLumen, 4)) + ', Mean: ' + str(np.round(avg_precisionLumen, 4)) + '  <-- Artery lumen')
    logger.info('SEGMENTATION RESULTS MEASURED BY OBJECT-LEVEL DICE SCORES:')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreTub, 4)) + ', Std: ' + str(np.round(std_dice_scoreTub, 4)) + ', Min: ' + str(np.round(min_dice_scoreTub, 4)) + ', Max: ' + str(np.round(max_dice_scoreTub, 4)) + '  <-- Tubuli')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreGlom, 4)) + ', Std: ' + str(np.round(std_dice_scoreGlom, 4)) + ', Min: ' + str(np.round(min_dice_scoreGlom, 4)) + ', Max: ' + str(np.round(max_dice_scoreGlom, 4)) + '  <-- Glomeruli (incl. tuft)')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreTuft, 4)) + ', Std: ' + str(np.round(std_dice_scoreTuft, 4)) + ', Min: ' + str(np.round(min_dice_scoreTuft, 4)) + ', Max: ' + str(np.round(max_dice_scoreTuft, 4)) + '  <-- Tuft')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreVeins, 4)) + ', Std: ' + str(np.round(std_dice_scoreVeins, 4)) + ', Min: ' + str(np.round(min_dice_scoreVeins, 4)) + ', Max: ' + str(np.round(max_dice_scoreVeins, 4)) + '  <-- Veins')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreArtery, 4)) + ', Std: ' + str(np.round(std_dice_scoreArtery, 4)) + ', Min: ' + str(np.round(min_dice_scoreArtery, 4)) + ', Max: ' + str(np.round(max_dice_scoreArtery, 4)) + '  <-- Artery (incl. lumen)')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreLumen, 4)) + ', Std: ' + str(np.round(std_dice_scoreLumen, 4)) + ', Min: ' + str(np.round(min_dice_scoreLumen, 4)) + ', Max: ' + str(np.round(max_dice_scoreLumen, 4)) + '  <-- Artery lumen')

    if saveResults:
        figPath = resultsPath + '/QuantitativeResults'
        if not os.path.exists(figPath):
            os.makedirs(figPath)

        disease = diseaseModels[evaluatorID // 2]

        np.save(figPath + '/' + disease + '_tubuliDice.npy', np.array(allClassEvaluators[evaluatorID][0].diceScores))
        np.save(figPath + '/' + disease + '_glomDice.npy', np.array(allClassEvaluators[evaluatorID][1].diceScores))
        np.save(figPath + '/' + disease + '_tuftDice.npy', np.array(allClassEvaluators[evaluatorID][2].diceScores))
        np.save(figPath + '/' + disease + '_veinsDice.npy', np.array(allClassEvaluators[evaluatorID][3].diceScores))
        np.save(figPath + '/' + disease + '_arteriesDice.npy', np.array(allClassEvaluators[evaluatorID][4].diceScores))
        np.save(figPath + '/' + disease + '_lumenDice.npy', np.array(allClassEvaluators[evaluatorID][5].diceScores))

        np.save(figPath + '/' + disease + '_detectionResults.npy', np.stack((precisionsTub, precisionsGlom, precisionsTuft, precisionsVeins, precisionsArtery, precisionsLumen)))


    if applyTestTimeAugmentation:
        precisionsTub, avg_precisionTub, avg_dice_scoreTub, std_dice_scoreTub, min_dice_scoreTub, max_dice_scoreTub = allClassEvaluators[evaluatorID+1][0].score()  # tubuliresults
        precisionsGlom, avg_precisionGlom, avg_dice_scoreGlom, std_dice_scoreGlom, min_dice_scoreGlom, max_dice_scoreGlom = allClassEvaluators[evaluatorID+1][1].score()  # tubuliresults
        precisionsTuft, avg_precisionTuft, avg_dice_scoreTuft, std_dice_scoreTuft, min_dice_scoreTuft, max_dice_scoreTuft = allClassEvaluators[evaluatorID+1][2].score()  # tubuliresults
        precisionsVeins, avg_precisionVeins, avg_dice_scoreVeins, std_dice_scoreVeins, min_dice_scoreVeins, max_dice_scoreVeins = allClassEvaluators[evaluatorID+1][3].score()  # tubuliresults
        precisionsArtery, avg_precisionArtery, avg_dice_scoreArtery, std_dice_scoreArtery, min_dice_scoreArtery, max_dice_scoreArtery = allClassEvaluators[evaluatorID+1][4].score()  # tubuliresults
        precisionsLumen, avg_precisionLumen, avg_dice_scoreLumen, std_dice_scoreLumen, min_dice_scoreLumen, max_dice_scoreLumen = allClassEvaluators[evaluatorID+1][5].score()  # tubuliresults
        logger.info('TTA DETECTION RESULTS MEASURED BY AVERAGE PRECISION:')
        logger.info('0.5    0.55    0.6    0.65    0.7    0.75    0.8    0.85    0.9 <- Thresholds')
        logger.info(str(np.round(precisionsTub, 4)) + ', Mean: ' + str(np.round(avg_precisionTub, 4)) + '  <-- Tubuli')
        logger.info(str(np.round(precisionsGlom, 4)) + ', Mean: ' + str(np.round(avg_precisionGlom, 4)) + '  <-- Glomeruli (incl. tuft)')
        logger.info(str(np.round(precisionsTuft, 4)) + ', Mean: ' + str(np.round(avg_precisionTuft, 4)) + '  <-- Tuft')
        logger.info(str(np.round(precisionsVeins, 4)) + ', Mean: ' + str(np.round(avg_precisionVeins, 4)) + '  <-- Veins')
        logger.info(str(np.round(precisionsArtery, 4)) + ', Mean: ' + str(np.round(avg_precisionArtery, 4)) + '  <-- Artery (incl. lumen)')
        logger.info(str(np.round(precisionsLumen, 4)) + ', Mean: ' + str(np.round(avg_precisionLumen, 4)) + '  <-- Artery lumen')
        logger.info('TTA SEGMENTATION RESULTS MEASURED BY OBJECT-LEVEL DICE SCORES:')
        logger.info('Mean: ' + str(np.round(avg_dice_scoreTub, 4)) + ', Std: ' + str(np.round(std_dice_scoreTub, 4)) + ', Min: ' + str(np.round(min_dice_scoreTub, 4)) + ', Max: ' + str(np.round(max_dice_scoreTub, 4)) + '  <-- Tubuli')
        logger.info('Mean: ' + str(np.round(avg_dice_scoreGlom, 4)) + ', Std: ' + str(np.round(std_dice_scoreGlom, 4)) + ', Min: ' + str(np.round(min_dice_scoreGlom, 4)) + ', Max: ' + str(np.round(max_dice_scoreGlom, 4)) + '  <-- Glomeruli (incl. tuft)')
        logger.info('Mean: ' + str(np.round(avg_dice_scoreTuft, 4)) + ', Std: ' + str(np.round(std_dice_scoreTuft, 4)) + ', Min: ' + str(np.round(min_dice_scoreTuft, 4)) + ', Max: ' + str(np.round(max_dice_scoreTuft, 4)) + '  <-- Tuft')
        logger.info('Mean: ' + str(np.round(avg_dice_scoreVeins, 4)) + ', Std: ' + str(np.round(std_dice_scoreVeins, 4)) + ', Min: ' + str(np.round(min_dice_scoreVeins, 4)) + ', Max: ' + str(np.round(max_dice_scoreVeins, 4)) + '  <-- Veins')
        logger.info('Mean: ' + str(np.round(avg_dice_scoreArtery, 4)) + ', Std: ' + str(np.round(std_dice_scoreArtery, 4)) + ', Min: ' + str(np.round(min_dice_scoreArtery, 4)) + ', Max: ' + str(np.round(max_dice_scoreArtery, 4)) + '  <-- Artery (incl. lumen)')
        logger.info('Mean: ' + str(np.round(avg_dice_scoreLumen, 4)) + ', Std: ' + str(np.round(std_dice_scoreLumen, 4)) + ', Min: ' + str(np.round(min_dice_scoreLumen, 4)) + ', Max: ' + str(np.round(max_dice_scoreLumen, 4)) + '  <-- Artery lumen')

        if saveResults:
            np.save(figPath + '/' + disease + '_tubuliDice_TTA.npy', np.array(allClassEvaluators[evaluatorID + 1][0].diceScores))
            np.save(figPath + '/' + disease + '_glomDice_TTA.npy', np.array(allClassEvaluators[evaluatorID + 1][1].diceScores))
            np.save(figPath + '/' + disease + '_tuftDice_TTA.npy', np.array(allClassEvaluators[evaluatorID + 1][2].diceScores))
            np.save(figPath + '/' + disease + '_veinsDice_TTA.npy', np.array(allClassEvaluators[evaluatorID + 1][3].diceScores))
            np.save(figPath + '/' + disease + '_arteriesDice_TTA.npy', np.array(allClassEvaluators[evaluatorID + 1][4].diceScores))
            np.save(figPath + '/' + disease + '_lumenDice_TTA.npy', np.array(allClassEvaluators[evaluatorID + 1][5].diceScores))

            np.save(figPath + '/' + disease + '_detectionResults_TTA.npy', np.stack((precisionsTub, precisionsGlom, precisionsTuft, precisionsVeins, precisionsArtery, precisionsLumen)))



def patchify(patches: np.ndarray, patch_size: Tuple[int, int], step: int = 1):
    return view_as_windows(patches, patch_size, step)

def unpatchify(patches: np.ndarray, imsize: Tuple[int, int]):

    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    # Calculat the overlap size in each axis
    o_w = (n_w * p_w - i_w) / (n_w - 1)
    o_h = (n_h * p_h - i_h) / (n_h - 1)

    # The overlap should be integer, otherwise the patches are unable to reconstruct into a image with given shape
    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor

# Examplary use:
# # # # # # # # #
# import numpy as np
# from patchify import patchify, unpatchify
#
# image = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
#
# patches = patchify(image, (2,2), step=1) # split image into 2*3 small 2*2 patches.
#
# assert patches.shape == (2, 3, 2, 2)
# reconstructed_image = unpatchify(patches, image.shape)
#
# assert (reconstructed_image == image).all()



def getChannelSmootingConvLayer(channels, kernel_size=5, sigma=1.5):

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          (-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)).float()
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def overlayVisualization(img, lbl, imgStr, lblStr):
    # remove tubuli border prepare visualization
    lbl[lbl==7] = 0
    customColorMap = mpl.colors.ListedColormap(['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'brown'])
    lblMasked = np.ma.masked_where(lbl == 0, lbl)
    sizeShift = (640-516)//2

    plt.figure(figsize=(20,8))
    plt.subplot(131)
    plt.imshow(img)
    plt.axis('off')
    plt.title(imgStr)
    plt.subplot(132)
    plt.imshow(lbl, cmap=customColorMap, vmin = 0, vmax = 7)
    plt.axis('off')
    plt.title(lblStr)
    plt.subplot(133)
    plt.imshow(img[sizeShift:sizeShift+516, sizeShift:sizeShift+516, :])
    plt.imshow(lblMasked, cmap=customColorMap, vmin = 0, vmax = 7, alpha=0.5)
    plt.axis('off')
    plt.title('Overlay')
    plt.subplots_adjust(wspace=0, hspace=0)
  
  
  
if __name__ == '__main__':
    print()
