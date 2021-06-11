import skimage.measure
import numpy as np

# this class evaluates prediction performance of a single class/label using instace dice scores as well as average precisions
class ClassEvaluator(object):

    def __init__(self, thres=None):
        self.thres = np.arange(start=0.5, stop=0.95, step=0.05) if thres is None else thres
        self.TP = np.zeros(self.thres.__len__())
        self.FN = np.zeros(self.thres.__len__())
        self.FP = np.zeros(self.thres.__len__())
        self.diceScores = []

    # add prediction/ground-truth pair
    def add_example(self, pred, gt):
        gtInstances = np.unique(gt)
        gt_num = len(gtInstances[gtInstances != 0])
        IoU_dict = []  # (prediction label)-(IoU)
        # match_dict = {}  # (prediction label)-(matched gt label)

        pred_area = self.get_area_dict(pred)
        gt_area = self.get_area_dict(gt)
        unique = np.unique(pred)

        # compute dice scores of each predicted instance with its maximally overlapping ground-truth instance
        for label in unique:
            if label == 0:
                continue
            u, c = np.unique(gt[pred == label], return_counts=True)
            ind = np.argsort(c, kind='mergesort')
            if len(u) == 1 and u[0] == 0: # only background contained
                IoU_dict.append(0)
                # match_dict[label] = None
                self.diceScores.append(0)
            else:
                # take the gt label with the largest overlap
                i = ind[-2] if u[ind[-1]] == 0 else ind[-1]
                intersect = c[i]
                union = pred_area[label] + gt_area[u[i]] - intersect
                IoU_dict.append(intersect / union)
                # match_dict[label] = u[i]
                diceScore = 2*intersect / (pred_area[label] + gt_area[u[i]])
                self.diceScores.append(diceScore)

        # count all TP, FP, FN in current image
        IoU_dict = np.array(IoU_dict)
        for i, threshold in enumerate(self.thres):
            tp = np.sum(IoU_dict > threshold)
            self.FP[i] += len(IoU_dict) - tp
            self.FN[i] += gt_num - tp
            self.TP[i] += tp

        # also compute dice scores of each ground-truth instance with its maximally overlapping prediction instance
        uniqueGT = np.unique(gt)
        for label in uniqueGT:
            if label == 0:
                continue
            u, c = np.unique(pred[gt == label], return_counts=True)
            ind = np.argsort(c, kind='mergesort')
            if len(u) == 1 and u[0] == 0:  # only background contained
                self.diceScores.append(0)
            else:
                # take the gt label with the largest overlap
                i = ind[-2] if u[ind[-1]] == 0 else ind[-1]
                intersect = c[i]
                diceScore = 2 * intersect / (gt_area[label] + pred_area[u[i]])
                self.diceScores.append(diceScore)

    # measure area regions
    def get_area_dict(self, label_map):
        props = skimage.measure.regionprops(label_map)
        return {p.label: p.area for p in props}

    # compute average precision for each threshold
    def score(self):
        precisions = self.TP / (self.TP + self.FN + self.FP)
        avg_precision = np.mean(precisions)

        avg_dice_score = np.mean(np.array(self.diceScores))
        std_dice_score = np.std(np.array(self.diceScores))
        min_dice_score = np.min(np.array(self.diceScores))
        max_dice_score = np.max(np.array(self.diceScores))

        return precisions, avg_precision, avg_dice_score, std_dice_score, min_dice_score, max_dice_score


if __name__ == "__main__":
    print('')

