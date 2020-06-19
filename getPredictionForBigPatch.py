import os
import sys
import numpy as np
import torch
import openslide as osl
import time
from skimage.transform import rescale, resize
from utils import patchify, savePredictionOverlayResults, savePredictionResultsWithoutDilation, saveImage, savePredictionResults, saveRGBPredictionOverlayResults, convert_labelmap_to_rgb_with_instance_first_class
from model import Custom
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_fill_holes
import logging as log

from utils import getChannelSmootingConvLayer

# This script can be used to automatically segment a selected rectangular patch within a specified WSI! #
# Type in path to specific WSI, deep learning model and results folder
WSIpath='<ABSOLUTE PATH TO WSI>'
modelpath = '<ABSOLUTE PATH TO TRAINED MODEL>'
resultspath = '<ABSOLUTE PATH TO RESULTS FOLDER>'

if not os.path.exists(resultspath):
    os.makedirs(resultspath)

start = time.time()

# specify device to apply network on
GPUno = 0
device = torch.device("cuda:" + str(GPUno) if torch.cuda.is_available() else "cpu")

# Apply postprocessing techniques:
# centerWeightingOfPrediction = 5
applyTestTimeAugmentation = True
applyPredictionSmooting = False
holefilling = True

# Settings designed for WSI-Segmentation
applyForegroundExtraction = False
savePredictionNumpy = False
loadPredictionNumpy = False

structure = np.zeros((3, 3), dtype=np.int)
structure[1, :] = 1
structure[:, 1] = 1

# Select raw coordinates (e.g. Qupath coordinates) of the left upper corner of the selected patch
patchCenterCoordinatesRaw = np.array([30954, 6375])

# specify how big the selected patch will be, [2, 2] represents 2*516 pixels width and 2*516 pixels height 
patchGridCellTimes = np.array([2, 2])
# when performing inference for a whole row within a patch, choose how often to split this row to prevent VRAM issues
gpuAmountRowPatchSplits = 4
# select on which porportion to move the sliding segmentation window
strideProportion = 1.0

spacings = np.array([float(slide.properties['openslide.mpp-x']), float(slide.properties['openslide.mpp-y'])])
sizeOfAnnotatedCellsResampled = np.array([516, 516])
sizeOfImageCellResampled = np.array([640, 640])

# type in how many pixels, depending on the WSI pixel sizes (spacing), would represent 174um, you can also automatically compute it by:
# sizeOfAnnotatedCells = np.asarray(np.round(174. / spacings), np.int32)
sizeOfAnnotatedCells = np.array([768, 768])


# load slide
slide = osl.OpenSlide(WSIpath)


patchCenterCoordinates = np.asarray(patchCenterCoordinatesRaw // spacings, np.int)

# load model
model = Custom(input_ch=3, output_ch=8, modelDim=2)

# state_dict = torch.load(modelpath, map_location=lambda storage, loc: storage)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:]  # remove `module.`
#     new_state_dict[name] = v
# # load params
# model.load_state_dict(new_state_dict)

model.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage))
model.train(False)
model.eval()
model = model.to(device)
# model = nn.DataParallel(model).to(device)

amountTotalClasses = 8

downsamplingFactor = sizeOfAnnotatedCellsResampled[0]/sizeOfAnnotatedCells[0]

sizeOfImageCell = np.asarray(np.round(sizeOfImageCellResampled / downsamplingFactor), np.int)

segmentationPatchStride = np.asarray(sizeOfAnnotatedCellsResampled * strideProportion, np.int)

sizeOfSegmentationPatch = np.asarray((1-strideProportion)*sizeOfAnnotatedCellsResampled, np.int) + patchGridCellTimes * segmentationPatchStride
sizeOfExtractedImagePatch = np.asarray(((sizeOfSegmentationPatch + sizeOfImageCellResampled - sizeOfAnnotatedCellsResampled) / downsamplingFactor), np.int)

figHeight = 10
figSize = (sizeOfSegmentationPatch[0]/sizeOfSegmentationPatch[1]*figHeight, figHeight)

# Set up logger
log.basicConfig(
    level=log.INFO,
    format='%(message)s',
    handlers=[
        log.FileHandler(resultspath + '/results.log', 'w'),
        log.StreamHandler(sys.stdout)
    ])
logger = log.getLogger()

logger.info(WSIpath)
logger.info(modelpath)
logger.info(resultspath)

logger.info('Apply test time augmentation: '+str(applyTestTimeAugmentation))
logger.info('Apply gaussian prediction smoothing: '+str(applyPredictionSmooting))
logger.info('PatchCenterCoordinates: '+str(patchCenterCoordinates))
logger.info('PatchGridCellTimes: '+str(patchGridCellTimes))
logger.info('SizeOfExtractedImagePatch: '+str(sizeOfExtractedImagePatch))
logger.info('SizeOfSegmentationPatch: '+str(sizeOfSegmentationPatch))

# save prediciton computation time when already computed by loading prediction and then postprocessing
if loadPredictionNumpy:
    extractedResampledBigPatch = np.load(resultspath + '/extractedResampledBigPatch.npy')
    finalBigPatchPrediction = np.load(resultspath + '/finalBigPatchPrediction.npy')
else:
    # extract patch from WSI
    extractedPatch = slide.read_region(location=patchCenterCoordinates, level=0, size=sizeOfExtractedImagePatch) 

    extractedPatch = np.array(extractedPatch)[:,:,:-1] #remove alpha channel

    # rescale patch according to our chosen pixel sizes
    extractedPatch = rescale(extractedPatch, downsamplingFactor, order=1, preserve_range=False, multichannel=True)
    logger.info(tuple(reversed(extractedPatch.shape[:2])))
    logger.info(sizeOfSegmentationPatch + sizeOfImageCellResampled - sizeOfAnnotatedCellsResampled)
    assert (tuple(reversed(extractedPatch.shape[:2])) == (sizeOfSegmentationPatch + sizeOfImageCellResampled - sizeOfAnnotatedCellsResampled)).all(), "Rescale leads to size problems...Need fixing"

    # preprocessing
    extractedPatchPre = (np.array(extractedPatch * 3.2 - 1.6, np.float32)).transpose(2, 0, 1) #initial 255.0 division unnecessary since preserve range normalizes to [0,1]

    # divide huge image patch into multiple smaller patches of sizes 640x640x3
    smallOverlappingPatches = patchify(extractedPatchPre.copy(), patch_size=(3,sizeOfImageCellResampled[0],sizeOfImageCellResampled[1]), step=segmentationPatchStride[0]) #shape: (1, 5, 7, 3, 512, 512)

    assert (tuple(reversed(smallOverlappingPatches.shape[1:3])) == patchGridCellTimes).all(), "Error...fix patchify result sizes"

    # Choose either storing data on RAM or VRAM:
    # smallOverlappingPatches = torch.from_numpy(smallOverlappingPatches).to(device)
    smallOverlappingPatches = torch.from_numpy(smallOverlappingPatches)

    # bigPatchResults = torch.zeros(device=device, size=(amountTotalClasses, sizeOfSegmentationPatch[1], sizeOfSegmentationPatch[0])) #shape: (8, 1536, 2048)
    bigPatchResults = torch.zeros(device="cpu", size=(amountTotalClasses, sizeOfSegmentationPatch[1], sizeOfSegmentationPatch[0])) #shape: (8, 1536, 2048)

    amountOfRowPatches = smallOverlappingPatches.size()[2]
    gpuIDXsplits = np.array_split(np.arange(amountOfRowPatches), gpuAmountRowPatchSplits)

    # compute prediction for each row, also for each split of a row and save predictions into 'bigPatchResults'
    for x in range(smallOverlappingPatches.size()[1]):
        for i in range(gpuAmountRowPatchSplits):
            imgBatch = smallOverlappingPatches[0, x, gpuIDXsplits[i], :, :, :].to(device)

            with torch.no_grad():
                rowPrediction = torch.softmax(model(imgBatch), dim=1) #shape: (7, 8, 512, 512)

                if applyTestTimeAugmentation:
                    imgBatch = imgBatch.flip(2)
                    rowPrediction += torch.softmax(model(imgBatch), 1).flip(2)

                    imgBatch = imgBatch.flip(3)
                    rowPrediction += torch.softmax(model(imgBatch), 1).flip(3).flip(2)

                    imgBatch = imgBatch.flip(2)
                    rowPrediction += torch.softmax(model(imgBatch), 1).flip(3)

                # # Center weighting
                # rowPrediction[:, :, patchSize//4 : patchSize//4*3, patchSize//4 : patchSize//4*3] *= 2

                rowPrediction = rowPrediction.to("cpu")

                for idx, y in enumerate(gpuIDXsplits[i]):
                    bigPatchResults[:, segmentationPatchStride[0]*x:sizeOfAnnotatedCellsResampled[0]+segmentationPatchStride[0]*x, segmentationPatchStride[1]*y:sizeOfAnnotatedCellsResampled[1]+segmentationPatchStride[1]*y] += rowPrediction[idx, :, :, :]


    # optional: smooth prediction probabilities using gaussian kernels
    with torch.no_grad():
        if applyPredictionSmooting:
            bigPatchResults = bigPatchResults.unsqueeze(0)
            smoothingKernel = getChannelSmootingConvLayer(8).to(device)
            bigPatchResults = smoothingKernel(bigPatchResults).squeeze(0)

        # compute final prediction label map
        finalBigPatchPrediction = torch.argmax(bigPatchResults, 0).to("cpu").numpy() #shape: (1536, 2048)

    # remove offset between image and lbl size
    offset = (sizeOfImageCellResampled[0]-sizeOfAnnotatedCellsResampled[0])//2
    extractedPatch = extractedPatch[offset : extractedPatch.shape[0] - offset, offset : extractedPatch.shape[1] - offset,:]
    extractedResampledBigPatch = np.asarray(np.round(extractedPatch * 255.), np.uint8) #shape: (1536, 2048, 3)

    assert extractedResampledBigPatch.shape[0:2] == bigPatchResults.shape[1:3], "Segmentation patch result size unequal overlapping rgb image"

    if savePredictionNumpy:
        np.save(resultspath + '/extractedResampledBigPatch.npy', extractedResampledBigPatch)
        np.save(resultspath + '/finalBigPatchPrediction.npy', finalBigPatchPrediction)


# save image and prediction in different modes:

saveImage(extractedResampledBigPatch, resultspath + '/Coord_'+str(patchCenterCoordinatesRaw[0])+'_'+str(patchCenterCoordinatesRaw[1])+'_OrigPatch.png', figSize)

# finalBigPatchPrediction[finalBigPatchPrediction == 7] = 0

savePredictionResultsWithoutDilation(finalBigPatchPrediction, resultspath + '/Coord_'+str(patchCenterCoordinatesRaw[0])+'_'+str(patchCenterCoordinatesRaw[1])+'_Prediction1_output.png', figSize)

# savePredictionOverlayResults(extractedResampledBigPatch, finalBigPatchPrediction, resultspath + '/Coord_'+str(patchCenterCoordinatesRaw[0])+'_'+str(patchCenterCoordinatesRaw[1])+'_Prediction3_anOverlay.png', figSize, alpha=0.4)

logger.info('########### POSTPROCESSING STARTS...###########')

finalBigPatchPrediction[finalBigPatchPrediction == 7] = 0


# ################# REMOVING TOO SMALL CONNECTED REGIONS ################
# Tuft
labeledTubuli, numberTubuli = label(np.asarray(finalBigPatchPrediction == 3, np.uint8), structure)  # datatype of 'labeledTubuli': int32
for i in range(1, numberTubuli + 1):
    tubuliSelection = (labeledTubuli == i)
    if tubuliSelection.sum() < 500:  # remove too small noisy regions
        finalBigPatchPrediction[tubuliSelection] = 2

# Glomeruli
labeledTubuli, numberTubuli = label(np.asarray(np.logical_or(finalBigPatchPrediction == 3, finalBigPatchPrediction == 2), np.uint8), structure)  # datatype of 'labeledTubuli': int32
for i in range(1, numberTubuli + 1):
    tubuliSelection = (labeledTubuli == i)
    if tubuliSelection.sum() < 1500:  # remove too small noisy regions
        finalBigPatchPrediction[tubuliSelection] = 0

# Artery lumen
labeledTubuli, numberTubuli = label(np.asarray(finalBigPatchPrediction == 6, np.uint8), structure)  # datatype of 'labeledTubuli': int32
for i in range(1, numberTubuli + 1):
    tubuliSelection = (labeledTubuli == i)
    if tubuliSelection.sum() < 20:  # remove too small noisy regions
        finalBigPatchPrediction[tubuliSelection] = 5

# Full artery
labeledTubuli, numberTubuli = label(np.asarray(np.logical_or(finalBigPatchPrediction == 5, finalBigPatchPrediction == 6), np.uint8), structure)  # datatype of 'labeledTubuli': int32
for i in range(1, numberTubuli + 1):
    tubuliSelection = (labeledTubuli == i)
    if tubuliSelection.sum() < 400:  # remove too small noisy regions
        finalBigPatchPrediction[tubuliSelection] = 0

# Veins
labeledTubuli, numberTubuli = label(np.asarray(finalBigPatchPrediction == 4, np.uint8), structure)  # datatype of 'labeledTubuli': int32
for i in range(1, numberTubuli + 1):
    tubuliSelection = (labeledTubuli == i)
    if tubuliSelection.sum() < 3000:  # remove too small noisy regions
        finalBigPatchPrediction[tubuliSelection] = 0

# Tubuli
labeledTubuli, numberTubuli = label(np.asarray(finalBigPatchPrediction == 1, np.uint8), structure)  # datatype of 'labeledTubuli': int32
for i in range(1, numberTubuli + 1):
    tubuliSelection = (labeledTubuli == i)
    if tubuliSelection.sum() < 400:  # remove too small noisy regions
        finalBigPatchPrediction[tubuliSelection] = 0



################# HOLE FILLING ################
if holefilling:
    finalBigPatchPrediction[binary_fill_holes(finalBigPatchPrediction == 1)] = 1  # tubuli
    finalBigPatchPrediction[binary_fill_holes(finalBigPatchPrediction == 4)] = 4  # veins
    tempTuftMask = binary_fill_holes(finalBigPatchPrediction == 3)  # tuft
    finalBigPatchPrediction[binary_fill_holes(np.logical_or(finalBigPatchPrediction == 3, finalBigPatchPrediction == 2))] = 2  # glom
    finalBigPatchPrediction[tempTuftMask] = 3  # tuft
    tempArteryLumenMask = binary_fill_holes(finalBigPatchPrediction == 6)  # artery_lumen
    finalBigPatchPrediction[binary_fill_holes(np.logical_or(finalBigPatchPrediction == 5, finalBigPatchPrediction == 6))] = 5  # full_artery
    finalBigPatchPrediction[tempArteryLumenMask] = 6  # artery_lumen

logger.info('########### POSTPROCESSING DONE...###########')

# convert prediction to rgb image
finalBigPatchPredictionRGB = convert_labelmap_to_rgb_with_instance_first_class(finalBigPatchPrediction, structure) #3 Channel-Dim. as last dim again...

savePredictionResults(finalBigPatchPrediction, resultspath + '/Coord_'+str(patchCenterCoordinatesRaw[0])+'_'+str(patchCenterCoordinatesRaw[1])+'_Prediction2_processed.png', figSize)

savePredictionOverlayResults(extractedResampledBigPatch, finalBigPatchPrediction, resultspath + '/Coord_'+str(patchCenterCoordinatesRaw[0])+'_'+str(patchCenterCoordinatesRaw[1])+'_Prediction3_anOverlay.png', figSize, alpha=0.4)

saveImage(finalBigPatchPredictionRGB, resultspath + '/Coord_'+str(patchCenterCoordinatesRaw[0])+'_'+str(patchCenterCoordinatesRaw[1])+'_Prediction4_Instance.png', figSize)
saveRGBPredictionOverlayResults(extractedResampledBigPatch, finalBigPatchPredictionRGB, resultspath + '/Coord_'+str(patchCenterCoordinatesRaw[0])+'_'+str(patchCenterCoordinatesRaw[1])+'_Prediction5_Instance_Overlay.png', figSize, alpha=0.4)

end = time.time()

logger.info('Run time: '+str(end-start)+' sec.')


