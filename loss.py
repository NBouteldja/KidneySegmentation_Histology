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
        Computes the dice loss (averaging dice scores across B x C) between network prediction and target used for training
        :param prediction: BxCxHxW (2d) or BxCxHxWxD (3d) float tensor, CARE: prediction results straight
               after last conv without being finally propagated through an activation (softmax, sigmoid)
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


class DiceLossGivenMask(nn.Module):
    def __init__(self):
        super(DiceLossGivenMask, self).__init__()
        self.eps = 1e-6
        self.smooth = 1.

    def forward(self, prediction, target, mask):
        """
        Fast dice loss computation when mask given
        :param prediction: predictions without activation function
        :param target: one-hot float tensor
        :param mask: float tensor of prediction size to ignore certain spatial predictions
        """

        predictionNorm = F.sigmoid(prediction)
        # predictionNorm = F.softmax(prediction, dim=1)

        denominator = predictionNorm + target
        denominator = denominator * mask

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


class PseudoDiceLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(PseudoDiceLoss, self).__init__()
        self.eps = 1e-6
        self.smooth = 1.
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        """
        Computes the pseudo dice loss (averaging dice scores across C) between network prediction and target used for training
        :param prediction: BxCxHxW (2d) or BxCxHxWxD (3d) float tensor, CARE: prediction results straight
               after last conv without being finally propagated through an activation (softmax, sigmoid)
        :param target: BxCxHxW (2d) or BxCxHxWxD (3d) float tensor representing the ground-truth as one-hot encoding
        :return: 1 - mean dice score across channel amount C
        """
        predictionNorm = F.sigmoid(prediction)
        if self.ignore_index != -1:
            mask = (target == self.ignore_index)
            target[mask] = 0

        if target.dtype == torch.int64:
            target = getOneHotEncoding(target)

        if self.ignore_index != -1:
            mask = mask.unsqueeze(1).expand_as(target)
            target[mask] = 0
            predictionNorm[mask] = 0

        if target.dim() == 4: #2D
            numerator = 2. * (predictionNorm * target).sum(3).sum(2).sum(0) + self.smooth
            denominator = (predictionNorm + target).sum(3).sum(2).sum(0) + self.eps + self.smooth
            dice_score = numerator / denominator
            return 1.0 - dice_score.mean()

        elif target.dim() == 5: #3D
            numerator = 2. * (predictionNorm * target).sum(4).sum(3).sum(2).sum(0) + self.smooth
            denominator = (predictionNorm + target).sum(4).sum(3).sum(2).sum(0) + self.eps + self.smooth
            dice_score = numerator / denominator
            return 1.0 - dice_score.mean()
        else:
            ValueError('Given data dimension >' + str(target.dim()) + 'd< not supported!')


class HingeDiceLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(HingeDiceLoss, self).__init__()
        self.eps = 1e-6
        self.smooth = 1.
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        """
        Computes the hinge dice loss (averaging dice scores across B x C) between network prediction and target used for training
        :param prediction: BxCxHxW (2d) or BxCxHxWxD (3d) float tensor, CARE: prediction results straight
               after last conv without being finally propagated through an activation (softmax, sigmoid)
        :param target: BxCxHxW (2d) or BxCxHxWxD (3d) float tensor representing the ground-truth as one-hot encoding
        :return: 1 - mean hinge dice score across BxC
        """
        predictionNorm = F.sigmoid(prediction)
        if self.ignore_index != -1:
            mask = (target == self.ignore_index)
            target[mask] = 0

        if target.dtype == torch.int64:
            target = getOneHotEncoding(target)

        if self.ignore_index != -1:
            mask = mask.unsqueeze(1).expand_as(target)
            target[mask] = 0
            predictionNorm[mask] = 0

        if target.dim() == 4: #2D
            numerator = 2. * (predictionNorm * target).sum(3).sum(2) + self.smooth
            denominator = (predictionNorm + target).sum(3).sum(2) + self.eps + self.smooth
            dice_score = numerator / denominator
            h1 = (torch.clamp(dice_score, max=0.1) * 10 - 1) ** 2
            h2 = (torch.clamp(dice_score, max=0.01) * 100 - 1) ** 2
            return 1.0 - dice_score.mean() + h1.mean()*10 + h2.mean()*10

        elif target.dim() == 5: #3D
            numerator = 2. * (predictionNorm * target).sum(4).sum(3).sum(2) + self.smooth
            denominator = (predictionNorm + target).sum(4).sum(3).sum(2) + self.eps + self.smooth
            dice_score = numerator / denominator
            h1 = (torch.clamp(dice_score, max=0.1) * 10 - 1) ** 2
            h2 = (torch.clamp(dice_score, max=0.01) * 100 - 1) ** 2
            return 1.0 - dice_score.mean() + h1.mean()*10 + h2.mean()*10
        else:
            ValueError('Given data dimension >' + str(target.dim()) + 'd< not supported!')


if __name__ == '__main__':
    print()