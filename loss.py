import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import getOneHotEncoding


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6
        self.smooth = 1.
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        """
        Computes the dice loss (by averaging dice scores across B x C) between network prediction and target used for training
        :param prediction: BxCxHxW (2d) or BxCxHxWxD (3d) float tensor, CARE: prediction results straight
               after last conv without being finally propagated through a final activation function (softmax, sigmoid)
        :param target: BxCxHxW (2d) or BxCxHxWxD (3d) float tensor representing the ground-truth as one-hot encoding
        :return: 1 - mean dice score across BxC
        """

        predictionNorm = F.sigmoid(prediction)
        # predictionNorm = F.softmax(prediction, dim=1)
        if self.ignore_index != -1:
            target = target.clone().detach()
            mask = target == self.ignore_index
            target[mask] = 0

        if target.dtype == torch.int64:
            target = getOneHotEncoding(prediction, target)

        if self.ignore_index != -1 and target.size()[1] != 1:
            mask = mask.unsqueeze(1).expand_as(target)
            target[mask] = 0

        denominator = predictionNorm + target
        if self.ignore_index != -1:
            denominator[mask] = 0

        if target.dim() == 4: #2D
            numerator = 2. * (predictionNorm * target).sum(3).sum(2) + self.smooth
            denominator = denominator.sum(3).sum(2) + self.eps + self.smooth
            dice_score = numerator / denominator
            return 1.0 - dice_score.mean()

        elif target.dim() == 5: #3D
            numerator = 2. * (predictionNorm * target).sum(4).sum(3).sum(2) + self.smooth
            denominator = denominator.sum(4).sum(3).sum(2) + self.eps + self.smooth
            dice_score = numerator / denominator
            return 1.0 - dice_score.mean()
        else:
            ValueError('Given data dimension >' + str(target.dim()) + 'd< not supported!')

if __name__ == '__main__':
    print()
