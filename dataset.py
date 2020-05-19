import os

import numpy as np
import torch

from PIL import Image
import cv2
# from skimage.io import imread
from skimage.transform import rotate
from scipy.ndimage.interpolation import zoom
# from sklearn.cluster import KMeans

from torch.utils.data import Dataset
from torchvision import transforms

from imgaug import augmenters as iaa
import imgaug as ia

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation

def preprocessingGT(lbl):
    structure = np.zeros((3, 3), dtype=np.int)
    structure[1, :] = 1
    structure[:, 1] = 1

    allGlomeruli = np.logical_or(lbl == 2, lbl == 3)
    labeledGlom, numberGlom = label(np.asarray(allGlomeruli, np.uint8), structure)
    temp = np.zeros(lbl.shape)
    for i in range(1, numberGlom + 1):
        temp += binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(labeledGlom == i)))))))
    glomBorder = np.logical_and(temp > 1, np.logical_not(allGlomeruli))
    lbl[binary_dilation(glomBorder)] = 7

    allArteries = np.logical_or(lbl == 5, lbl == 6)
    labeledGlom, numberGlom = label(np.asarray(allArteries, np.uint8), structure)
    temp = np.zeros(lbl.shape)
    for i in range(1, numberGlom + 1):
        temp += binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(labeledGlom == i)))))
    glomBorder = np.logical_and(temp > 1, np.logical_not(allArteries))
    lbl[binary_dilation(glomBorder)] = 7


class CustomDataSetRAM(Dataset):
    def __init__(self, datasetType, logger):
        self.transformIMG = None
        self.transformLBL = None
        self.transform_WhenNoAugm = transforms.Compose([
            RangeNormaliziation(),
            ToTensor()
        ])

        self.data = []
        self.useAugm = datasetType=='train'
        self.lblShape = 0


        if self.useAugm:
            self.transformIMG, self.transformLBL = get_Augmentation_Transf()
            logger.info('Augmentation method:')
            logger.info(self.transformIMG)

        assert datasetType in ['train', 'val', 'test'], '### ERROR: WRONG DATASETTYPE '+datasetType+' INPUTTED ! ###'

        image_dir_base = '<PATH-TO-DATA-FOLDER>'
        if datasetType == 'train':
            image_dir = image_dir_base + '/Train'
        elif datasetType == 'val':
            image_dir = image_dir_base + '/Val'
        elif datasetType == 'test':
            image_dir = image_dir_base + '/Test'

        label_dir = image_dir
        files = sorted(list(filter(lambda x: ').png' in x, os.listdir(image_dir))))

        logger.info('Loading dataset with size: '+str(len(files)))
        for k, fname in enumerate(files):
            imagePath = os.path.join(image_dir, fname)
            labelPath = os.path.join(label_dir, fname.replace('.png', '-labels.png'))

            img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
            img = img[:640, :640, :]
            lbl = np.array(Image.open(labelPath))

            preprocessingGT(lbl)

            logger.info("Load data with index " + str(k) + " : " + fname + ", ImgShape: " + str(img.shape) + " " + str(img.dtype) + ", LabelShape: " + str(lbl.shape) + " " + str(lbl.dtype) + " (max: " + str(lbl.max()) + ", min: " + str(lbl.min()) + ")")

            self.lblShape = lbl.shape
            if img.shape[:2] != lbl.shape:
                lbl = np.pad(lbl, ((img.shape[0]-lbl.shape[0])//2,(img.shape[1]-lbl.shape[1])//2), 'constant', constant_values=(8,8))

            self.data.append((img, lbl))

        assert len(files) > 0, 'No files found in ' + image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.useAugm:
            ia.seed(np.random.get_state()[1][0])

            img, lbl = self.data[index]

            seq_img_d = self.transformIMG.to_deterministic()
            seq_lbl_d = self.transformLBL.to_deterministic()

            seq_lbl_d = seq_lbl_d.copy_random_state(seq_img_d, matching="name")

            augmentedIMG = seq_img_d.augment_image(img)
            augmentedLBL = seq_lbl_d.augment_image(lbl)[(img.shape[0]-self.lblShape[0])//2:(img.shape[0]-self.lblShape[0])//2+self.lblShape[0],(img.shape[1]-self.lblShape[1])//2:(img.shape[1]-self.lblShape[1])//2+self.lblShape[1]]

            return self.transform_WhenNoAugm((augmentedIMG, augmentedLBL.copy()))
        else:
            img, lbl = self.data[index]
            return self.transform_WhenNoAugm((img, lbl[(img.shape[0]-self.lblShape[0])//2:(img.shape[0]-self.lblShape[0])//2+self.lblShape[0],(img.shape[1]-self.lblShape[1])//2:(img.shape[1]-self.lblShape[1])//2+self.lblShape[1]]))





class RandomHueSaturationValue(object):
    def __init__(self, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), p=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.p = p

    def __call__(self, sample):
        img, lbl = sample
        if np.random.random() < self.p:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(img)
            hue_shift = np.random.randint(self.hue_shift_limit[0], self.hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            img = cv2.merge((h, s, v))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        return img, lbl


class RandomRotation(object):
    def __init__(self, max_deg=20):
        self.max_deg = max_deg

    def __call__(self, sample):
        img, lbl = sample
        angle = np.random.uniform(low=-self.max_deg, high=self.max_deg)
        img = rotate(img, angle, preserve_range=True, mode='constant')
        lbl = rotate(lbl, angle, order=0, preserve_range=True, mode='constant')
        return img, lbl


# CARE: Changes image sizes!!
class RandomScaling(object):
    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, sample):
        img, lbl = sample
        perc = np.random.uniform(low=1.-self.factor, high=1.+self.factor)
        img = zoom(img, zoom=perc, mode='constant', cval=0)
        lbl = zoom(lbl, zoom=perc, order=0, mode='constant', cval=0)
        return img, lbl


class RandomShiftScaleRotate(object):
    def __init__(self, shift_limit, scale_limit, rotate_limit, aspect_limit, p):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.aspect_limit = aspect_limit
        self.p = p

    def __call__(self, sample):
        img, lbl = sample

        if np.random.random() < self.p:
            height, width, channel = img.shape

            angle = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1])
            scale = np.random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1])
            aspect = np.random.uniform(1 + self.aspect_limit[0], 1 + self.aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * width)
            dy = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)

            img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))
            lbl = cv2.warpPerspective(lbl, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))

        return img, lbl


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, lbl = sample
        if np.random.uniform() < self.p:
            img = cv2.flip(img, 1)
            lbl = cv2.flip(lbl, 1)
        return img, lbl

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, lbl = sample
        if np.random.uniform() < self.p:
            img = cv2.flip(img, 0)
            lbl = cv2.flip(lbl, 0)
        return img, lbl


class CenterCrop(object):
    def __init__(self, size=(192, 192)):
        self.size = size

    def __call__(self, sample):
        img, lbl = sample
        dx = (img.shape[0]-self.size[0]) // 2
        dy = (img.shape[1]-self.size[1]) // 2
        img = img[dx:dx+self.size[0], dy:dy+self.size[1], :]
        lbl = lbl[dx:dx+self.size[0], dy:dy+self.size[1], :]
        return img, lbl


class RangeNormaliziation(object):
    def __call__(self, sample):
        img, lbl = sample
        return img / 255.0 * 3.2 - 1.6, lbl


class ToTensor(object):
    def __call__(self, sample):
        img, lbl = sample

        lbl = torch.from_numpy(lbl).long()
        img = torch.from_numpy(np.array(img, np.float32).transpose(2, 0, 1))

        return img, lbl



def get_Augmentation_Transf():
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.

    sometimes = lambda aug: iaa.Sometimes(0.5, aug, name="Random1")
    sometimes2 = lambda aug: iaa.Sometimes(0.2, aug, name="Random2")
    sometimes3 = lambda aug: iaa.Sometimes(0.9, aug, name="Random3")
    sometimes4 = lambda aug: iaa.Sometimes(0.9, aug, name="Random4")
    sometimes5 = lambda aug: iaa.Sometimes(0.9, aug, name="Random5")

    seq_img = iaa.Sequential([
        iaa.AddToHueAndSaturation(value=(-13, 13), name="MyHSV"),
        sometimes2(iaa.GammaContrast(gamma=(0.85, 1.15), name="MyGamma")),
        iaa.Fliplr(0.5, name="MyFlipLR"),
        iaa.Flipud(0.5, name="MyFlipUD"),
        sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
        iaa.OneOf([
            sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=0, name="MyPiece")),
            sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=0, name="MyElastic")),
            sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=0, name="MyAffine"))
        ], name="MyOneOf")
    ], name="MyAug")

    seq_lbl = iaa.Sequential([
        iaa.Fliplr(0.5, name="MyFlipLR"),
        iaa.Flipud(0.5, name="MyFlipUD"),
        sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
        iaa.OneOf([
            sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=8, order=0, name="MyPiece")),
            sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=8, order=0, name="MyElastic")),
            sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=8, order=0, name="MyAffine"))
        ], name="MyOneOf")
    ], name="MyAug")

    return seq_img, seq_lbl



if '__main__' == __name__:
    print()