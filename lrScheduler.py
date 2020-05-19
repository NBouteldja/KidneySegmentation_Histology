import torch


LR_Reduce_No_Train_Improvement = 15
LR_Reduce_No_Val_Improvement = 15
EARLY_STOP_LR_TOO_LOW = 4e-6


class MyLRScheduler():
    def __init__(self, optimizer, model, foldResultsModelPath, setting, initLR, divideLRfactor):
        self.optimizer = optimizer
        self.model = model
        self.foldResultsModelPath = foldResultsModelPath
        self.currentLR = initLR
        self.divideLRfactor = divideLRfactor

        self.noImprovement = 0

        if 'val' in setting:
            self.bestValue = -1
        else:
            self.bestValue = 1E4


    def stepTrain(self, newTrainLoss, logger):
        # Update learning rate
        if newTrainLoss >= self.bestValue:
            self.noImprovement += 1

            if self.noImprovement >= LR_Reduce_No_Train_Improvement:
                self.model.load_state_dict(torch.load(self.foldResultsModelPath + '/currentBestTrainModel.pt'))
                self.update_lr_by_divison(self.divideLRfactor)
                logger.info('### After '+str(LR_Reduce_No_Train_Improvement)+' no train loss reduction => Best model loaded and LR reduced to '+str(self.currentLR)+' !')
                if self.currentLR < EARLY_STOP_LR_TOO_LOW:
                    return True
                self.noImprovement = 0
        else:
            self.noImprovement = 0
            self.bestValue = newTrainLoss
            torch.save(self.model.state_dict(), self.foldResultsModelPath + '/currentBestTrainModel.pt')

        return False


    def stepTrainVal(self, newValScore, logger):
        # Update learning rate
        if newValScore <= self.bestValue:
            self.noImprovement += 1

            if self.noImprovement >= LR_Reduce_No_Val_Improvement:
                self.model.load_state_dict(torch.load(self.foldResultsModelPath + '/currentBestValModel.pt'))
                self.update_lr_by_divison(self.divideLRfactor)
                logger.info('### After ' + str(LR_Reduce_No_Val_Improvement) + ' no val score improvement => Best model loaded and LR reduced to ' + str(self.currentLR) + ' !')
                if self.currentLR < EARLY_STOP_LR_TOO_LOW:
                    return True
                self.noImprovement = 0
        else:
            self.noImprovement = 0
            self.bestValue = newValScore
            torch.save(self.model.state_dict(), self.foldResultsModelPath + '/currentBestValModel.pt')

        return False


    def update_lr_by_divison(self, factor):
        newLR = self.currentLR / factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = newLR
        self.currentLR = newLR


    def loadBestValIntoModel(self):
        self.model.load_state_dict(torch.load(self.foldResultsModelPath + '/currentBestValModel.pt'))
