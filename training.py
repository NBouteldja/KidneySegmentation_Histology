import os
import shutil
import numpy as np
import logging as log
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from RAdam import RAdam
from dataset import CustomDataSetRAM
from model import Custom, UNetVanillaMod, CE_mod
from utils import getCrossValSplits, parse_nvidia_smi, parse_RAM_info, countParam, getDiceScores, getMeanDiceScores, convert_labelmap_to_rgb, saveFigureResults, printResultsForDiseaseModel
from loss import DiceLoss
from lrScheduler import MyLRScheduler
from postprocessing import postprocessPredictionAndGT, extractInstanceChannels
from evaluation import ClassEvaluator

import warnings
warnings.filterwarnings("ignore")



#################################### General GPU settings ####################################
GPUno = 0
useAllAvailableGPU = False
device = torch.device("cuda:" + str(GPUno) if torch.cuda.is_available() else "cpu")
############################### Dataset length specification #################################
                    # Healthy, UUO, Adenine, IRI, NTN
testDatasetsSizes = np.array([160, 60, 60, 60, 60, 60])
diseaseModels = ['HEALTHY', 'UUO', 'ADENINE', 'ALPORT', 'IRI', 'NTN']
##################################### Save test results ######################################
saveFinalTestResults = True
############################### Apply Test Time Augmentation #################################
applyTestTimeAugmentation = True
##############################################################################################


# this method trains a network with the given specification
def train(model, setting, optimizer, scheduler, epochs, batchSize, logger, resultsPath, testResults, testResultsTTA, tbWriter, allClassEvaluators):

    model.to(device)
    if torch.cuda.device_count() > 1 and useAllAvailableGPU:
        logger.info('# {} GPUs utilized! #'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # mandatory to produce random numpy numbers during training, otherwise batches will contain equal random numbers (originally: numpy issue)
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # allocate and separately load train / val / test data sets
    dataset_Train = CustomDataSetRAM('train', logger)
    dataloader_Train = DataLoader(dataset=dataset_Train, batch_size = batchSize, shuffle = True, num_workers = 4, worker_init_fn=worker_init_fn)

    if 'val' in setting:
        dataset_Val = CustomDataSetRAM('val', logger)
        dataloader_Val = DataLoader(dataset=dataset_Val, batch_size = batchSize, shuffle = False, num_workers = 1, worker_init_fn=worker_init_fn)

    if 'test' in setting:
        dataset_Test = CustomDataSetRAM('test', logger)
        dataloader_Test = DataLoader(dataset=dataset_Test, batch_size = batchSize, shuffle = False, num_workers = 1, worker_init_fn=worker_init_fn)

    logger.info('####### DATA LOADED - TRAINING STARTS... #######')

    # Utilize dice loss and weighted cross entropy loss
    Dice_Loss = DiceLoss(ignore_index=8).to(device)
    CE_Loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1., 1., 1., 1., 1., 1., 1., 10.]), ignore_index=8).to(device)
    # WCE_Loss = nn.CrossEntropyLoss(weight=getWeightsForCEloss(dataset, train_idx, areLabelsOnehotEncoded=False, device=device, logger=logger)).to(device)

    for epoch in range(epochs):
        model.train(True)

        epochCELoss = 0
        epochDiceLoss = 0
        epochLoss = 0

        np.random.seed()
        start = time.time()
        for batch in dataloader_Train:
            # get data and put onto device
            imgBatch, segBatch = batch

            imgBatch = imgBatch.to(device)
            segBatch = segBatch.to(device)

            optimizer.zero_grad()

            # forward image batch, compute loss and backprop
            prediction = model(imgBatch)

            CEloss = CE_Loss(prediction, segBatch)
            diceLoss = Dice_Loss(prediction, segBatch)

            loss = CEloss + diceLoss

            epochCELoss += CEloss.item()
            epochDiceLoss += diceLoss.item()
            epochLoss += loss.item()

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()

        epochTrainLoss = epochLoss / dataloader_Train.__len__()

        end = time.time()
        # print current loss
        logger.info('[Epoch '+str(epoch+1)+'] Train-Loss: '+str(round(epochTrainLoss,5))+', DiceLoss: '+
                    str(round(epochDiceLoss/dataloader_Train.__len__(),5))+', CELoss: '+str(round(epochCELoss/dataloader_Train.__len__(),5))+'  [took '+str(round(end-start,3))+'s]')

        # use tensorboard for visualization of training progress
        tbWriter.add_scalars('Plot/train', {'loss' : epochTrainLoss,
                                           'CEloss' : epochCELoss/dataloader_Train.__len__(),
                                           'DiceLoss' : epochDiceLoss/dataloader_Train.__len__()}, epoch)

        # each 50th epoch add prediction image to tensorboard
        if epoch % 50 == 0:
            with torch.no_grad():
                tbWriter.add_image('Train/_img', torch.round((imgBatch[0,:,:,:] + 1.6) / 3.2 * 255.0).byte() , epoch)
                tbWriter.add_image('Train/GT', convert_labelmap_to_rgb(segBatch[0,:,:].cpu()), epoch)
                tbWriter.add_image('Train/pred', convert_labelmap_to_rgb(prediction[0,:,:,:].argmax(0).cpu()), epoch)

        if epoch % 100 == 0:
            logger.info('[Epoch ' + str(epoch + 1) + '] ' + parse_nvidia_smi(GPUno))
            logger.info('[Epoch ' + str(epoch + 1) + '] ' + parse_RAM_info())


        # if validation is active, compute dice scores on validation data
        if 'val' in setting:
            model.train(False)

            diceScores_Val = []

            start = time.time()
            for batch in dataloader_Val:
                imgBatch, segBatch = batch
                imgBatch = imgBatch.to(device)
                # segBatch = segBatch.to(device)

                with torch.no_grad():
                    prediction = model(imgBatch).to('cpu')

                    diceScores_Val.append(getDiceScores(prediction, segBatch))

            diceScores_Val = np.concatenate(diceScores_Val, 0) # <- all dice scores of val data (batchSize x amountClasses-1)
            diceScores_Val = diceScores_Val[:, :-1]  # ignore last coloum=border dice scores

            mean_DiceScores_Val, epoch_val_mean_score = getMeanDiceScores(diceScores_Val, logger)

            end = time.time()
            logger.info('[Epoch '+str(epoch+1)+'] Val-Score (mean label dice scores): '+str(np.round(mean_DiceScores_Val,4))+', Mean: '+str(round(epoch_val_mean_score,4))+'  [took '+str(round(end-start,3))+'s]')

            tbWriter.add_scalar('Plot/val', epoch_val_mean_score, epoch)

            if epoch % 50 == 0:
                with torch.no_grad():
                    tbWriter.add_image('Val/_img', torch.round((imgBatch[0,:,:,:] + 1.6) / 3.2 * 255.0).byte(), epoch)
                    tbWriter.add_image('Val/GT', convert_labelmap_to_rgb(segBatch[0, :, :].cpu()), epoch)
                    tbWriter.add_image('Val/pred', convert_labelmap_to_rgb(prediction[0, :, :, :].argmax(0).cpu()), epoch)

            if epoch % 100 == 0:
                logger.info('[Epoch ' + str(epoch + 1) + ' - After Validation] ' + parse_nvidia_smi(GPUno))
                logger.info('[Epoch ' + str(epoch + 1) + ' - After Validation] ' + parse_RAM_info())


        # scheduler.step()
        if 'val' in setting:
            endLoop = scheduler.stepTrainVal(epoch_val_mean_score, logger)
        else:
            endLoop = scheduler.stepTrain(epochTrainLoss, logger)

        if epoch == (epochs - 1): #when no early stop is performed, load bestValModel into current model for later save
            logger.info('### No early stop performed! Best val model loaded... ####')
            if 'val' in setting:
                scheduler.loadBestValIntoModel()
                
        # if test is active, compute global dice scores on test data for fast and coarse performance check
        if 'test' in setting:
            model.train(False)

            diceScores_Test = []

            start = time.time()
            for batch in dataloader_Test:
                imgBatch, segBatch = batch
                imgBatch = imgBatch.to(device)
                # segBatch = segBatch.to(device)

                with torch.no_grad():
                    prediction = model(imgBatch).to('cpu')

                    diceScores_Test.append(getDiceScores(prediction, segBatch))


            diceScores_Test = np.concatenate(diceScores_Test, 0)  # <- all dice scores of test data (amountTestData x amountClasses-1)
            diceScores_Test = diceScores_Test[:,:-1] #ignore last coloum=border dice scores

            mean_DiceScores_Test, test_mean_score = getMeanDiceScores(diceScores_Test, logger)

            end = time.time()
            logger.info('[Epoch ' + str(epoch + 1) + '] Test-Score (mean label dice scores): ' + str(np.round(mean_DiceScores_Test, 4))+
                        ', Mean: ' + str(round(test_mean_score, 4)) + '  [took ' + str(round(end - start, 3)) + 's]')

            tbWriter.add_scalar('Plot/test', test_mean_score, epoch)

            if epoch % 50 == 0:
                with torch.no_grad():
                    tbWriter.add_image('Test/_img', torch.round((imgBatch[0,:,:,:] + 1.6) / 3.2 * 255.0).byte(), epoch)
                    tbWriter.add_image('Test/GT', convert_labelmap_to_rgb(segBatch[0, :, :].cpu()), epoch)
                    tbWriter.add_image('Test/pred', convert_labelmap_to_rgb(prediction[0, :, :, :].argmax(0).cpu()), epoch)

            if epoch % 100 == 0:
                logger.info('[Epoch ' + str(epoch + 1) + ' - After Testing] ' + parse_nvidia_smi(GPUno))
                logger.info('[Epoch ' + str(epoch + 1) + ' - After Testing] ' + parse_RAM_info())

            with torch.no_grad():
                ### if training is over ###
                if endLoop or (epoch == epochs - 1):

                    diceScores_Test = []
                    diceScores_Test_TTA = []

                    test_idx = np.arange(sum(testDatasetsSizes))
                    for sampleNo in test_idx:
                        diseaseID = -1
                        if sampleNo < sum(testDatasetsSizes[:1]):
                            diseaseID = 0 # Healthy test sample
                        elif sampleNo < sum(testDatasetsSizes[:2]):
                            diseaseID = 2 # UUO test sample
                        elif sampleNo < sum(testDatasetsSizes[:3]):
                            diseaseID = 4 # Adenine test sample
                        elif sampleNo < sum(testDatasetsSizes[:4]):
                            diseaseID = 6 # Alport test sample
                        elif sampleNo < sum(testDatasetsSizes[:5]):
                            diseaseID = 8 # IRI test sample
                        elif sampleNo < sum(testDatasetsSizes[:6]):
                            diseaseID = 10 # NTN test sample

                        # get test sample, forward it through network in evaluation mode, and compute performance
                        imgBatch, segBatch = dataset_Test.__getitem__(sampleNo)

                        imgBatch = imgBatch.unsqueeze(0).to(device)
                        segBatch = segBatch.unsqueeze(0)

                        prediction = model(imgBatch)

                        predictionCPU = prediction.to("cpu")

                        # apply post-processing
                        postprocessedPrediction, outputPrediction, preprocessedGT = postprocessPredictionAndGT(prediction, segBatch.squeeze(0).numpy(), device=device, predictionsmoothing=True, holefilling=True)
                        classInstancePredictionList, classInstanceGTList, finalPredictionRGB, preprocessedGTrgb = extractInstanceChannels(postprocessedPrediction, preprocessedGT, tubuliDilation=True)

                        # evaluate performance (TP, NP, FP counting and dice score computation)
                        for i in range(6): #number classes to evaluate = 6
                            allClassEvaluators[diseaseID][i].add_example(classInstancePredictionList[i],classInstanceGTList[i])

                        # compute global dice scores as coarse performance check
                        diceScores_Test.append(getDiceScores(predictionCPU, segBatch))


                        if saveFinalTestResults:
                            figFolder = resultsPath + '/' + diseaseModels[diseaseID // 2]
                            if not os.path.exists(figFolder):
                                os.makedirs(figFolder)

                            imgBatchCPU = torch.round((imgBatch[0, :, :, :].to("cpu") + 1.6) / 3.2 * 255.0).byte().numpy().transpose(1, 2, 0)
                            figPath = figFolder + '/test_idx_' + str(sampleNo) + '_result.png'
                            saveFigureResults(imgBatchCPU, outputPrediction, postprocessedPrediction, finalPredictionRGB, segBatch.squeeze(0).numpy(), preprocessedGT, preprocessedGTrgb, fullResultPath=figPath, alpha=0.4)

                        # perform exactly the same when applying TTA 
                        if applyTestTimeAugmentation:
                            prediction = torch.softmax(prediction, 1)

                            imgBatch = imgBatch.flip(2)
                            prediction += torch.softmax(model(imgBatch), 1).flip(2)

                            imgBatch = imgBatch.flip(3)
                            prediction += torch.softmax(model(imgBatch), 1).flip(3).flip(2)

                            imgBatch = imgBatch.flip(2)
                            prediction += torch.softmax(model(imgBatch), 1).flip(3)

                            prediction /= 4.

                            predictionCPU = prediction.to("cpu")

                            postprocessedPrediction, outputPrediction, preprocessedGT = postprocessPredictionAndGT(prediction, segBatch.squeeze(0).numpy(), device=device, predictionsmoothing=False,holefilling=True)

                            classInstancePredictionList, classInstanceGTList, finalPredictionRGB, preprocessedGTrgb = extractInstanceChannels(postprocessedPrediction, preprocessedGT, tubuliDilation=False)

                            for i in range(6):
                                allClassEvaluators[diseaseID+1][i].add_example(classInstancePredictionList[i], classInstanceGTList[i])

                            diceScores_Test_TTA.append(getDiceScores(predictionCPU, segBatch))

                            if saveFinalTestResults:
                                figPath = figFolder + '/test_idx_' + str(sampleNo) + '_result_TTA.png'
                                saveFigureResults(imgBatchCPU, outputPrediction, postprocessedPrediction, finalPredictionRGB, segBatch.squeeze(0).numpy(), preprocessedGT, preprocessedGTrgb, fullResultPath=figPath, alpha=0.4)


                    # print global dice scores as coarse performance check
                    diceScores_Test = np.concatenate(diceScores_Test, 0)  # <- all dice scores of test data (amountTestData x amountClasses-1)
                    diceScores_Test = diceScores_Test[:, :-1]  # ignore last coloum=border dice scores
                    mean_DiceScores_Test, test_mean_score = getMeanDiceScores(diceScores_Test, logger)
                    logger.info('[FINAL RESULT] [Epoch ' + str(epoch + 1) + '] Test-Score (mean label dice scores): ' + str(np.round(mean_DiceScores_Test, 4)) + ', Mean: ' + str(round(test_mean_score, 4)))
                    testResults.append(diceScores_Test)

                    # print global dice scores of TTA as coarse performance check
                    if applyTestTimeAugmentation:
                        diceScores_Test_TTA = np.concatenate(diceScores_Test_TTA, 0)  # <- all dice scores of test data (amountTestData x amountClasses-1)
                        diceScores_Test_TTA = diceScores_Test_TTA[:, :-1]  # ignore last coloum=border dice scores
                        mean_DiceScores_Test_TTA, test_mean_score_TTA = getMeanDiceScores(diceScores_Test_TTA, logger)
                        logger.info('[FINAL TTA RESULT] [Epoch '+str(epoch + 1)+'] Test-Score (mean label dice scores): '+str(np.round(mean_DiceScores_Test_TTA, 4)) + ', Mean: ' + str(round(test_mean_score_TTA, 4)))
                        testResultsTTA.append(diceScores_Test_TTA)


        if endLoop:
            logger.info('### Early network training stop at epoch '+str(epoch+1)+'! ###')
            break


    logger.info('[Epoch '+str(epoch+1)+'] ### Training done! ###')

    return model


def set_up_training(modelString, setting, epochs, batchSize, lrate, weightDecay, logger, resultsPath):

    logger.info('### SETTING -> {} <- ###'.format(setting.upper()))

    testResults = []
    testResultsTTA = []

    # 12*6 Evaluators: Healthy, Healthy_TTA, UUO, UUO_TTA, AdN, AdN_TTA, Alport, Alport_TTA, IRI, IRI_TTA, NTN, NTN_TTA
    allClassEvaluators = []
    for z in range(12):
        allClassEvaluators.append([ClassEvaluator(), ClassEvaluator(), ClassEvaluator(), ClassEvaluator(), ClassEvaluator(), ClassEvaluator()])
    
    resultsModelPath = resultsPath +'/Model'
    if not os.path.exists(resultsModelPath):
        os.makedirs(resultsModelPath)

    # setting up tensorboard visualization
    tensorboardPath = resultsPath + '/TB'
    shutil.rmtree(tensorboardPath, ignore_errors=True) #<- remove existing TB events
    tbWriter = SummaryWriter(log_dir=tensorboardPath)

    # load specified neural network
    if modelString == 'custom':
        model = Custom(input_ch=3, output_ch=8, modelDim=2)
    elif modelString == 'unet_mod':
        model = UNetVanillaMod(input_ch=3, output_ch=8, modelDim=2)
    elif modelString == 'ce_mod':
        model = CE_mod(input_ch=3, output_ch=8, modelDim=2)
    else:
        raise ValueError('Given model >' + modelString + '< is invalid!')

    logger.info('Model capacity: {} parameters.'.format(countParam(model)))

    # optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=weightDecay)
    optimizer = RAdam(model.parameters(), lr=lrate, weight_decay=weightDecay)

    scheduler = MyLRScheduler(optimizer, model, resultsModelPath, setting, initLR=lrate, divideLRfactor=3.0)

    # train network given specified specifications
    trained_model = train(
        model,
        setting,
        optimizer,
        scheduler,
        epochs,
        batchSize,
        logger,
        resultsPath,
        testResults,
        testResultsTTA,
        tbWriter,
        allClassEvaluators
    )

    torch.save(trained_model.state_dict(), resultsModelPath + '/finalModel.pt')

    if 'test' in setting:
        # print prediction performance results (instance-dice scores and Average Precisions) for each disease model!
        logger.info('### FINAL TEST RESULTS ###')
        allDiceScores = np.concatenate(testResults, 0)
        # Remove dice scores for border class
        allDiceScores = allDiceScores[:, :-1]
        logger.info('All test data foreground label dice scores:')
        logger.info(str(np.round(allDiceScores,4)))
        logger.info('Saving these dice scores in .csv file...')
        np.savetxt(resultsPath + '/allTestDiceScores.csv', allDiceScores, delimiter=',')

        meanLabelScores, meanOverallScores = getMeanDiceScores(allDiceScores, logger)

        logger.info('Mean Overall Foreground Label Dice Scores: '+str(np.round(meanLabelScores, 4)))
        logger.info('Mean Overall Dice Score: '+str(np.round(meanOverallScores, 4)))

        sampleDiceMeans = np.ma.masked_where(allDiceScores == -1, allDiceScores).mean(1).data
        sampleArgMax = sampleDiceMeans.argmax()
        sampleMeanDiceMax = sampleDiceMeans[sampleArgMax]
        sampleArgMin = sampleDiceMeans.argmin()
        sampleMeanDiceMin = sampleDiceMeans[sampleArgMin]

        logger.info('Sample with highest prediction performance (mean dice score: '+str(round(sampleMeanDiceMax,4))+') has index: '+str(sampleArgMax))
        logger.info('Sample with lowest prediction performance (mean dice score: '+str(round(sampleMeanDiceMin,4))+') has index: '+str(sampleArgMin))

        if applyTestTimeAugmentation:
            # print prediction performance TTA results (instance-dice scores and Average Precisions) for each disease model!
            allDiceScores_TTA = np.concatenate(testResultsTTA, 0)
            # Remove dice scores for border class
            allDiceScores_TTA = allDiceScores_TTA[:,:-1]
            logger.info('### FINAL TEST TIME AUGMENTATION RESULTS ###')
            logger.info('All test data foreground label dice scores (TTA):')
            logger.info(str(np.round(allDiceScores_TTA, 4)))
            logger.info('Saving these dice scores in .csv file...')
            np.savetxt(resultsPath + '/allTestDiceScores_TTA.csv', allDiceScores_TTA, delimiter=',')

            meanLabelScores_TTA, meanOverallScores_TTA = getMeanDiceScores(allDiceScores_TTA, logger)

            logger.info('Mean Overall Foreground Label Dice Scores (TTA): ' + str(np.round(meanLabelScores_TTA, 4)))
            logger.info('Mean Overall Dice Score (TTA): ' + str(np.round(meanOverallScores_TTA, 4)))

            sampleDiceMeans = np.ma.masked_where(allDiceScores_TTA == -1, allDiceScores_TTA).mean(1).data
            sampleArgMax = sampleDiceMeans.argmax()
            sampleMeanDiceMax = sampleDiceMeans[sampleArgMax]
            sampleArgMin = sampleDiceMeans.argmin()
            sampleMeanDiceMin = sampleDiceMeans[sampleArgMin]

            logger.info('Sample with highest TTA prediction performance (mean dice score: ' + str(round(sampleMeanDiceMax, 4)) + ') has index: ' + str(sampleArgMax))
            logger.info('Sample with lowest TTA prediction performance (mean dice score: ' + str(round(sampleMeanDiceMin, 4)) + ') has index: ' + str(sampleArgMin))

        # print quantitative performance results for each disease model
        for m in range(len(diseaseModels)):
            logger.info('############################### RESULTS FOR '+ diseaseModels[m] +' ###############################')
            printResultsForDiseaseModel(evaluatorID=2*m, allClassEvaluators=allClassEvaluators, applyTestTimeAugmentation=applyTestTimeAugmentation, logger=logger, saveResults=True, resultsPath=resultsPath, diseaseModels=diseaseModels)




if '__main__' == __name__:
    import argparse
    # parse arguments
    parser = argparse.ArgumentParser(description='python training.py -m <model-type> -d <dataset> -s <train_valid_test> -e <epochs> '+
                                                 '-b <batch-size> -r <learning-rate> -w <weight-decay>')
    parser.add_argument('-m', '--model', default='custom')
    parser.add_argument('-s', '--setting', default='train_val_test')
    parser.add_argument('-e', '--epochs', default=500, type=int)
    parser.add_argument('-b', '--batchSize', default=6, type=int)
    parser.add_argument('-r', '--lrate', default=0.001, type=float)
    parser.add_argument('-w', '--weightDecay', default=0.00001, type=float)

    options = parser.parse_args()
    assert(options.model in ['custom', 'unet_mod', 'ce_mod'])
    assert(options.setting in ['train_val_test', 'train_test', 'train_val', 'train'])
    assert(options.epochs > 0)
    assert(options.batchSize > 0)
    assert(options.lrate > 0)
    assert(options.weightDecay > 0)

    # Specify here the results path
    resultsPath = '<PATH-TO-RESULTS-FOLDER>'

    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    # Set up logger
    log.basicConfig(
        level=log.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        handlers=[
            log.FileHandler(resultsPath + '/LOGS.log','w'),
            log.StreamHandler(sys.stdout)
        ])
    logger = log.getLogger()

    logger.info('###### STARTED PROGRAM WITH OPTIONS: {} ######'.format(str(options)))

    torch.backends.cudnn.benchmark = True

    try:
        # start whole training and evaluation procedure
        set_up_training(modelString=options.model,
                                 setting=options.setting,
                                 epochs=options.epochs,
                                 batchSize=options.batchSize,
                                 lrate=options.lrate,
                                 weightDecay=options.weightDecay,
                                 logger=logger,
                                 resultsPath=resultsPath)
    except:
        logger.exception('! Exception !')
        raise

    log.info('%%%% Ended regularly ! %%%%')


