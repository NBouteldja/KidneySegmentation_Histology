import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial

import math


####################################### XAVIER WEIGHT INIT #########################################
def init_weights_xavier_normal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)



def getXandYgradTensor(featureChannels):
    return xGradientPre.repeat(featureChannels,1,1,1), yGradientPre.repeat(featureChannels,1,1,1)

def getGaussianKernel(featureChannels):
    return gaussian_kernel.repeat(featureChannels, 1, 1, 1)

xGradientPre = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view((1,1,3,3)).to("cuda:0")
yGradientPre = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]).view((1,1,3,3)).to("cuda:0")


kernel_size=5
sigma = 1.0
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
gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to("cuda:0")


nonlinearity = partial(F.relu, inplace=True)

################################### VARIOUS SEGMENTATION MODELS ####################################


####################################################################################################
# ----- Custom Unet 2D/3D - Pooling-Encoder + (Transposed/Upsampling)-Decoder + DoubleConvs ----- #
class Custom(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(Custom, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)

        self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False, conv5=False)
        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class Custom_768_512(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(Custom_768_512, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)
        self.down6 = down(1024, 1024, modelDim)

        self.up00 = up(1024, 1024, 1024, modelDim, upsampling=False)
        self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False, conv5=True)
        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x = self.up00(x7, x6)
        x = self.up0(x, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x

class CustomPretrainedEncoder(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomPretrainedEncoder, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.up0 = upSparse(512+4, 256, 256, modelDim, upsampling=False)
        self.up1 = upSparse(256, 128, 128, modelDim, upsampling=False)
        self.up2 = upSparse(128, 64, 64, modelDim, upsampling=False)
        self.up3 = upSparse(64, 64, 64, modelDim, upsampling=False)
        # self.up0 = up(512+4, 256, 256, modelDim, upsampling=False)
        # self.up1 = up(256, 128, 128, modelDim, upsampling=False)
        # self.up2 = up(128, 64, 64, modelDim, upsampling=False)
        # self.up3 = up(64, 64, 64, modelDim, upsampling=False)

        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalconv3 = nn.Conv2d(32, output_ch, 3)

        self.dacblock = DACblock(512, modelDim=2, withAtrous=True)
        self.rmpblock = RMPblock(512, modelDim=2)

        self.apply(init_weights_xavier_uniform)

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1 # 3 x (conv-bn-relu-conv-bn)-blocks
        self.encoder2 = resnet.layer2 # 4 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder3 = resnet.layer3 # 6 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder4 = resnet.layer4 # 3 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block


    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x4 = self.dacblock(x4)
        x4 = self.rmpblock(x4)

        x4 = self.up0(x4, x3)
        x4 = self.up1(x4, x2)
        x4 = self.up2(x4, x1)
        x4 = self.up3(x4, x)

        x4 = nonlinearity(self.finaldeconv1(x4))
        x4 = nonlinearity(self.finalconv2(x4))
        x4 = self.finalconv3(x4)

        # x4 = self.up4(x4, x1)
        #
        # x4 = self.outc(x4)

        return x4


class CustomPretrainedEncoderCustomized(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomPretrainedEncoderCustomized, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.init1 = nn.Conv2d(input_ch, 32, 3, padding=1)
        self.init2 = nn.BatchNorm2d(32)
        self.init3 = nn.ReLU()

        self.dacblock = DACblock(512, modelDim=2, withAtrous=True)
        self.rmpblock = RMPblock(512, modelDim=2)

        # self.up0 = upSparse(512, 256, 256, modelDim, upsampling=False)
        # self.up1 = upSparse(256, 128, 128, modelDim, upsampling=False)
        # self.up2 = upSparse(128, 64, 64, modelDim, upsampling=False)
        # self.up3 = upSparse(64, 64, 64, modelDim, upsampling=False)
        # self.up4 = upSparse(64, 32, 32, modelDim, upsampling=False)
        self.up0 = up(512+4, 256, 256, modelDim, upsampling=False)
        self.up1 = up(256, 128, 128, modelDim, upsampling=False)
        self.up2 = up(128, 64, 64, modelDim, upsampling=False)
        self.up3 = up(64, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False)

        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1 # 3 x (conv-bn-relu-conv-bn)-blocks
        self.encoder2 = resnet.layer2 # 4 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder3 = resnet.layer3 # 6 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder4 = resnet.layer4 # 3 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block


    def forward(self, x0):
        # Encoder
        x = self.firstconv(x0)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x4 = self.dacblock(x4)
        x4 = self.rmpblock(x4)

        x4 = self.up0(x4, x3)
        x4 = self.up1(x4, x2)
        x4 = self.up2(x4, x1)
        x4 = self.up3(x4, x)

        x4 = self.up4(x4, self.init3(self.init2(self.init1(x0))))

        x4 = self.outc(x4)

        return x4


class CustomPretrainedEncoderCustomizedContext1(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomPretrainedEncoderCustomizedContext1, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.init1 = nn.Conv2d(input_ch, 32, 3, padding=1)
        self.init2 = nn.BatchNorm2d(32)
        self.init3 = nn.ReLU()

        ftAmount = 512

        self.convAtrous1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=1, padding=1)
        self.normAtrous1 = nn.BatchNorm2d(ftAmount)
        self.convAtrous1_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=1, padding=1)
        self.normAtrous1_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=2, padding=2)
        self.normAtrous2 = nn.BatchNorm2d(ftAmount)
        self.convAtrous2_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=2, padding=2)
        self.normAtrous2_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=3, padding=3)
        self.normAtrous3 = nn.BatchNorm2d(ftAmount)
        self.convAtrous3_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=3, padding=3)
        self.normAtrous3_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=4, padding=4)
        self.normAtrous4 = nn.BatchNorm2d(ftAmount)
        self.convAtrous4_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=4, padding=4)
        self.normAtrous4_2 = nn.BatchNorm2d(ftAmount//4)


        self.convLearnedPool1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=2, stride=2, groups=ftAmount)
        self.normLearnedPool1 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, stride=3, groups=ftAmount)
        self.normLearnedPool2 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, stride=5, groups=ftAmount)
        self.normLearnedPool3 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=7, stride=7, groups=ftAmount)
        self.normLearnedPool4 = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Pool = nn.Conv2d(ftAmount*4, ftAmount, kernel_size=1)
        self.norm1x1_Pool = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Both = nn.Conv2d(ftAmount*2, ftAmount, kernel_size=1)
        self.norm1x1_Both = nn.BatchNorm2d(ftAmount)



        self.up0 = up(512, 256, 256, modelDim, upsampling=False)
        self.up1 = up(256, 128, 128, modelDim, upsampling=False)
        self.up2 = up(128, 64, 64, modelDim, upsampling=False)
        self.up3 = up(64, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False)

        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1 # 3 x (conv-bn-relu-conv-bn)-blocks
        self.encoder2 = resnet.layer2 # 4 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder3 = resnet.layer3 # 6 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder4 = resnet.layer4 # 3 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block


    def forward(self, x0):
        # Encoder
        x = self.firstconv(x0)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        in_channels, h, w = x4.size(1), x4.size(2), x4.size(3)

        ###### ATROUS CONV ######
        xAtrous1 = nonlinearity(self.normAtrous1_2(self.convAtrous1_2(nonlinearity(self.normAtrous1(self.convAtrous1(x4))))))
        xAtrous2 = nonlinearity(self.normAtrous2_2(self.convAtrous2_2(nonlinearity(self.normAtrous2(self.convAtrous2(x4))))))
        xAtrous3 = nonlinearity(self.normAtrous3_2(self.convAtrous3_2(nonlinearity(self.normAtrous3(self.convAtrous3(x4))))))
        xAtrous4 = nonlinearity(self.normAtrous4_2(self.convAtrous4_2(nonlinearity(self.normAtrous4(self.convAtrous4(x4))))))


        ######## POOLING ########
        xPool1 = nonlinearity(self.normLearnedPool1(self.convLearnedPool1(x4)))
        xPool2 = nonlinearity(self.normLearnedPool2(self.convLearnedPool2(x4)))
        xPool3 = nonlinearity(self.normLearnedPool3(self.convLearnedPool3(x4)))
        xPool4 = nonlinearity(self.normLearnedPool4(self.convLearnedPool4(x4)))

        xPool1 = F.upsample(xPool1, size=(h, w), mode='bilinear')
        xPool2 = F.upsample(xPool2, size=(h, w), mode='bilinear')
        xPool3 = F.upsample(xPool3, size=(h, w), mode='bilinear')
        xPool4 = F.upsample(xPool4, size=(h, w), mode='bilinear')

        xPool1 = nonlinearity(self.norm1x1_Pool(self.conv1x1_Pool(torch.cat((xPool1, xPool2, xPool3, xPool4), dim=1))))
        #########################

        x4 = nonlinearity(self.norm1x1_Both(self.conv1x1_Both(torch.cat((xAtrous1, xAtrous2, xAtrous3, xAtrous4, xPool1), dim=1))))


        x4 = self.up0(x4, x3)
        x4 = self.up1(x4, x2)
        x4 = self.up2(x4, x1)
        x4 = self.up3(x4, x)

        x4 = self.up4(x4, self.init3(self.init2(self.init1(x0))))

        x4 = self.outc(x4)

        return x4

class CustomPretrainedEncoderCustomizedContext(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomPretrainedEncoderCustomizedContext, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.init1 = nn.Conv2d(input_ch, 32, 3, padding=1)
        self.init2 = nn.BatchNorm2d(32)
        self.init3 = nn.ReLU()

        ftAmount = 128

        self.conv1x1_Down = nn.Conv2d(4*ftAmount, ftAmount, kernel_size=1)
        self.norm1x1_Down = nn.BatchNorm2d(ftAmount)

        self.convAtrous1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=1, padding=1)
        self.normAtrous1 = nn.BatchNorm2d(ftAmount)
        self.convAtrous1_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=1, padding=1)
        self.normAtrous1_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=2, padding=2)
        self.normAtrous2 = nn.BatchNorm2d(ftAmount)
        self.convAtrous2_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=2, padding=2)
        self.normAtrous2_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=3, padding=3)
        self.normAtrous3 = nn.BatchNorm2d(ftAmount)
        self.convAtrous3_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=3, padding=3)
        self.normAtrous3_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=4, padding=4)
        self.normAtrous4 = nn.BatchNorm2d(ftAmount)
        self.convAtrous4_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=4, padding=4)
        self.normAtrous4_2 = nn.BatchNorm2d(ftAmount//4)


        self.convLearnedPool1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=2, stride=2, groups=ftAmount)
        self.normLearnedPool1 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, stride=3, groups=ftAmount)
        self.normLearnedPool2 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, stride=5, groups=ftAmount)
        self.normLearnedPool3 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=7, stride=7, groups=ftAmount)
        self.normLearnedPool4 = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Pool = nn.Conv2d(ftAmount*4, ftAmount, kernel_size=1)
        self.norm1x1_Pool = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Both = nn.Conv2d(ftAmount*2, ftAmount*4, kernel_size=1)
        self.norm1x1_Both = nn.BatchNorm2d(ftAmount*4)


        self.up0 = up(512, 256, 256, modelDim, upsampling=False)
        self.up1 = up(256, 128, 128, modelDim, upsampling=False)
        self.up2 = up(128, 64, 64, modelDim, upsampling=False)
        self.up3 = up(64, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False)

        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1 # 3 x (conv-bn-relu-conv-bn)-blocks
        self.encoder2 = resnet.layer2 # 4 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder3 = resnet.layer3 # 6 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder4 = resnet.layer4 # 3 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block


    def forward(self, x0):
        # Encoder
        x = self.firstconv(x0)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        in_channels, h, w = x4.size(1), x4.size(2), x4.size(3)

        xC = nonlinearity(self.norm1x1_Down(self.conv1x1_Down(x4)))

        ###### ATROUS CONV ######
        xAtrous1 = nonlinearity(self.normAtrous1_2(self.convAtrous1_2(nonlinearity(self.normAtrous1(self.convAtrous1(xC))))))
        xAtrous2 = nonlinearity(self.normAtrous2_2(self.convAtrous2_2(nonlinearity(self.normAtrous2(self.convAtrous2(xC))))))
        xAtrous3 = nonlinearity(self.normAtrous3_2(self.convAtrous3_2(nonlinearity(self.normAtrous3(self.convAtrous3(xC))))))
        xAtrous4 = nonlinearity(self.normAtrous4_2(self.convAtrous4_2(nonlinearity(self.normAtrous4(self.convAtrous4(xC))))))


        ######## POOLING ########
        xPool1 = nonlinearity(self.normLearnedPool1(self.convLearnedPool1(xC)))
        xPool2 = nonlinearity(self.normLearnedPool2(self.convLearnedPool2(xC)))
        xPool3 = nonlinearity(self.normLearnedPool3(self.convLearnedPool3(xC)))
        xPool4 = nonlinearity(self.normLearnedPool4(self.convLearnedPool4(xC)))

        xPool1 = F.upsample(xPool1, size=(h, w), mode='bilinear')
        xPool2 = F.upsample(xPool2, size=(h, w), mode='bilinear')
        xPool3 = F.upsample(xPool3, size=(h, w), mode='bilinear')
        xPool4 = F.upsample(xPool4, size=(h, w), mode='bilinear')

        xPool1 = nonlinearity(self.norm1x1_Pool(self.conv1x1_Pool(torch.cat((xPool1, xPool2, xPool3, xPool4), dim=1))))
        #########################

        x4 = nonlinearity(self.norm1x1_Both(self.conv1x1_Both(torch.cat((xAtrous1, xAtrous2, xAtrous3, xAtrous4, xPool1), dim=1)))) + x4


        x4 = self.up0(x4, x3)
        x4 = self.up1(x4, x2)
        x4 = self.up2(x4, x1)
        x4 = self.up3(x4, x)

        x4 = self.up4(x4, self.init3(self.init2(self.init1(x0))))

        x4 = self.outc(x4)

        return x4


class CustomHourglass(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomHourglass, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)

        self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False)

        self.convF1 = nn.Conv2d(32, 32, kernel_size=3)
        self.normF1 = nn.InstanceNorm2d(32)
        self.reluF1 = nn.LeakyReLU()
        self.convO1 = nn.Conv2d(32, output_ch, kernel_size=1)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3)
        self.norm1 = nn.InstanceNorm2d(32)
        self.relu1 = nn.LeakyReLU()
        self.convB1 = nn.Conv2d(output_ch, 32, kernel_size=1)

        self.convF2 = nn.Conv2d(32, 32, kernel_size=3)
        self.normF2 = nn.InstanceNorm2d(32)
        self.reluF2 = nn.LeakyReLU()
        self.convO2 = nn.Conv2d(32, output_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.norm2 = nn.InstanceNorm2d(32)
        self.relu2 = nn.LeakyReLU()
        self.convB2 = nn.Conv2d(output_ch, 32, kernel_size=1)

        self.convF3 = nn.Conv2d(32, 32, kernel_size=3)
        self.normF3 = nn.InstanceNorm2d(32)
        self.reluF3 = nn.LeakyReLU()
        self.convO3 = nn.Conv2d(32, output_ch, kernel_size=1)


        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.reluF1(self.normF1(self.convF1(x)))

        x1 = self.convO1(x)

        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.relu1(self.norm1(self.conv1(x)))

        x = x + self.convB1(x1)

        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.reluF2(self.normF2(self.convF2(x)))

        x2 = self.convO2(x)

        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.relu2(self.norm2(self.conv2(x)))

        x = x + self.convB2(x1)

        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.reluF3(self.normF3(self.convF3(x)))

        x = self.convO3(x)

        return [x, x2, x1]


class ContextBlock(nn.Module):
    def __init__(self, channel):
        super(ContextBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out



class CustomContext(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomContext, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)

        # self.dacblock = DACblock(1024, modelDim=2, withAtrous=True)
        # self.rmpblock = RMPblock(1024, modelDim=2)

        ftAmount = 256

        self.conv1x1_Down = nn.Conv2d(4*ftAmount, ftAmount, kernel_size=1)
        self.norm1x1_Down = nn.BatchNorm2d(ftAmount)

        self.convAtrous1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, dilation=1, padding=2)
        self.normAtrous1 = nn.BatchNorm2d(ftAmount)

        self.convAtrous2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, dilation=2, padding=4)
        self.normAtrous2 = nn.BatchNorm2d(ftAmount)

        self.convAtrous3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, dilation=3, padding=6)
        self.normAtrous3 = nn.BatchNorm2d(ftAmount)

        self.convAtrous4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, dilation=4, padding=8)
        self.normAtrous4 = nn.BatchNorm2d(ftAmount)


        self.convLearnedPool1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=2, stride=2, groups=ftAmount)
        self.normLearnedPool1 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, stride=3, groups=ftAmount)
        self.normLearnedPool2 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, stride=5, groups=ftAmount)
        self.normLearnedPool3 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=7, stride=7, groups=ftAmount)
        self.normLearnedPool4 = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Pool = nn.Conv2d(ftAmount*4, ftAmount, kernel_size=1)
        self.norm1x1_Pool = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Atrous = nn.Conv2d(ftAmount*4, ftAmount, kernel_size=1)
        self.norm1x1_Atrous = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Both = nn.Conv2d(ftAmount*2, ftAmount*4, kernel_size=1)
        self.norm1x1_Both = nn.BatchNorm2d(ftAmount*4)

        # self.up0 = up(1024+4, 512, 512, modelDim, upsampling=False)
        self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False, conv5=False)
        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        # x6 = self.dacblock(x6)
        # x6 = self.rmpblock(x6)

        in_channels, h, w = x6.size(1), x6.size(2), x6.size(3)

        xC = nonlinearity(self.norm1x1_Down(self.conv1x1_Down(x6)))

        ###### ATROUS CONV ######
        xAtrous1 = nonlinearity(self.normAtrous1(self.convAtrous1(xC)))
        xAtrous2 = nonlinearity(self.normAtrous2(self.convAtrous2(xC)))
        xAtrous3 = nonlinearity(self.normAtrous3(self.convAtrous3(xC)))
        xAtrous4 = nonlinearity(self.normAtrous4(self.convAtrous4(xC)))

        xAtrous1 = nonlinearity(self.norm1x1_Atrous(self.conv1x1_Atrous(torch.cat((xAtrous1, xAtrous2, xAtrous3, xAtrous4), dim=1))))

        ######## POOLING ########
        xPool1 = nonlinearity(self.normLearnedPool1(self.convLearnedPool1(xC)))
        xPool2 = nonlinearity(self.normLearnedPool2(self.convLearnedPool2(xC)))
        xPool3 = nonlinearity(self.normLearnedPool3(self.convLearnedPool3(xC)))
        xPool4 = nonlinearity(self.normLearnedPool4(self.convLearnedPool4(xC)))

        xPool1 = F.upsample(xPool1, size=(h, w), mode='bilinear')
        xPool2 = F.upsample(xPool2, size=(h, w), mode='bilinear')
        xPool3 = F.upsample(xPool3, size=(h, w), mode='bilinear')
        xPool4 = F.upsample(xPool4, size=(h, w), mode='bilinear')

        xPool1 = nonlinearity(self.norm1x1_Pool(self.conv1x1_Pool(torch.cat((xPool1, xPool2, xPool3, xPool4), dim=1))))
        #########################

        x6 = nonlinearity(self.norm1x1_Both(self.conv1x1_Both(torch.cat((xAtrous1, xPool1), dim=1)))) + x6

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class CustomContext_768_512(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomContext_768_512, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)

        self.convMid = nn.Conv2d(1024, 1024, kernel_size=5)
        self.normMid = nn.InstanceNorm2d(1024)
        self.reluMid = nn.LeakyReLU(inplace=True)

        # self.dacblock = DACblock(1024, modelDim=2, withAtrous=True)
        # self.rmpblock = RMPblock(1024, modelDim=2)

        ftAmount = 256

        self.conv1x1_Down = nn.Conv2d(4*ftAmount, ftAmount, kernel_size=1)
        self.norm1x1_Down = nn.BatchNorm2d(ftAmount)

        self.convAtrous1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, dilation=1, padding=2)
        self.normAtrous1 = nn.BatchNorm2d(ftAmount)

        self.convAtrous2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, dilation=2, padding=4)
        self.normAtrous2 = nn.BatchNorm2d(ftAmount)

        self.convAtrous3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, dilation=3, padding=6)
        self.normAtrous3 = nn.BatchNorm2d(ftAmount)

        self.convAtrous4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, dilation=4, padding=8)
        self.normAtrous4 = nn.BatchNorm2d(ftAmount)


        self.convLearnedPool1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=2, stride=2, groups=ftAmount)
        self.normLearnedPool1 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, stride=3, groups=ftAmount)
        self.normLearnedPool2 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, stride=5, groups=ftAmount)
        self.normLearnedPool3 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=7, stride=7, groups=ftAmount)
        self.normLearnedPool4 = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Pool = nn.Conv2d(ftAmount*4, ftAmount, kernel_size=1)
        self.norm1x1_Pool = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Atrous = nn.Conv2d(ftAmount*4, ftAmount, kernel_size=1)
        self.norm1x1_Atrous = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Both = nn.Conv2d(ftAmount*2, ftAmount*4, kernel_size=1)
        self.norm1x1_Both = nn.BatchNorm2d(ftAmount*4)

        # self.up0 = up(1024+4, 512, 512, modelDim, upsampling=False)
        self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False, conv5=True)
        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x6 = self.reluMid(self.normMid(self.convMid(x6)))

        # x6 = self.dacblock(x6)
        # x6 = self.rmpblock(x6)

        in_channels, h, w = x6.size(1), x6.size(2), x6.size(3)

        xC = nonlinearity(self.norm1x1_Down(self.conv1x1_Down(x6)))

        ###### ATROUS CONV ######
        xAtrous1 = nonlinearity(self.normAtrous1(self.convAtrous1(xC)))
        xAtrous2 = nonlinearity(self.normAtrous2(self.convAtrous2(xC)))
        xAtrous3 = nonlinearity(self.normAtrous3(self.convAtrous3(xC)))
        xAtrous4 = nonlinearity(self.normAtrous4(self.convAtrous4(xC)))

        xAtrous1 = nonlinearity(self.norm1x1_Atrous(self.conv1x1_Atrous(torch.cat((xAtrous1, xAtrous2, xAtrous3, xAtrous4), dim=1))))

        ######## POOLING ########
        xPool1 = nonlinearity(self.normLearnedPool1(self.convLearnedPool1(xC)))
        xPool2 = nonlinearity(self.normLearnedPool2(self.convLearnedPool2(xC)))
        xPool3 = nonlinearity(self.normLearnedPool3(self.convLearnedPool3(xC)))
        xPool4 = nonlinearity(self.normLearnedPool4(self.convLearnedPool4(xC)))

        xPool1 = F.upsample(xPool1, size=(h, w), mode='bilinear')
        xPool2 = F.upsample(xPool2, size=(h, w), mode='bilinear')
        xPool3 = F.upsample(xPool3, size=(h, w), mode='bilinear')
        xPool4 = F.upsample(xPool4, size=(h, w), mode='bilinear')

        xPool1 = nonlinearity(self.norm1x1_Pool(self.conv1x1_Pool(torch.cat((xPool1, xPool2, xPool3, xPool4), dim=1))))
        #########################

        x6 = nonlinearity(self.norm1x1_Both(self.conv1x1_Both(torch.cat((xAtrous1, xPool1), dim=1)))) + x6

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class CustomContext3(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomContext3, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)

        # self.dacblock = DACblock(1024, modelDim=2, withAtrous=True)
        # self.rmpblock = RMPblock(1024, modelDim=2)

        ftAmount = 256

        self.conv1x1_Down = nn.Conv2d(4*ftAmount, ftAmount, kernel_size=1)
        self.norm1x1_Down = nn.BatchNorm2d(ftAmount)

        self.convAtrous1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=1, padding=1)
        self.normAtrous1 = nn.BatchNorm2d(ftAmount)
        self.convAtrous1_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=1, padding=1)
        self.normAtrous1_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=2, padding=2)
        self.normAtrous2 = nn.BatchNorm2d(ftAmount)
        self.convAtrous2_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=2, padding=2)
        self.normAtrous2_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=3, padding=3)
        self.normAtrous3 = nn.BatchNorm2d(ftAmount)
        self.convAtrous3_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=3, padding=3)
        self.normAtrous3_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=4, padding=4)
        self.normAtrous4 = nn.BatchNorm2d(ftAmount)
        self.convAtrous4_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=4, padding=4)
        self.normAtrous4_2 = nn.BatchNorm2d(ftAmount//4)


        self.convLearnedPool1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=2, stride=2, groups=ftAmount)
        self.normLearnedPool1 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, stride=3, groups=ftAmount)
        self.normLearnedPool2 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, stride=5, groups=ftAmount)
        self.normLearnedPool3 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=7, stride=7, groups=ftAmount)
        self.normLearnedPool4 = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Pool = nn.Conv2d(ftAmount*4, ftAmount, kernel_size=1)
        self.norm1x1_Pool = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Atrous = nn.Conv2d(ftAmount, ftAmount, kernel_size=1)
        self.norm1x1_Atrous = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Both = nn.Conv2d(ftAmount*2, ftAmount*4, kernel_size=1)
        self.norm1x1_Both = nn.BatchNorm2d(ftAmount*4)

        # self.up0 = up(1024+4, 512, 512, modelDim, upsampling=False)
        self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False)
        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        # x6 = self.dacblock(x6)
        # x6 = self.rmpblock(x6)

        in_channels, h, w = x6.size(1), x6.size(2), x6.size(3)

        xC = nonlinearity(self.norm1x1_Down(self.conv1x1_Down(x6)))

        ###### ATROUS CONV ######
        xAtrous1 = nonlinearity(self.normAtrous1_2(self.convAtrous1_2(nonlinearity(self.normAtrous1(self.convAtrous1(xC))))))
        xAtrous2 = nonlinearity(self.normAtrous2_2(self.convAtrous2_2(nonlinearity(self.normAtrous2(self.convAtrous2(xC))))))
        xAtrous3 = nonlinearity(self.normAtrous3_2(self.convAtrous3_2(nonlinearity(self.normAtrous3(self.convAtrous3(xC))))))
        xAtrous4 = nonlinearity(self.normAtrous4_2(self.convAtrous4_2(nonlinearity(self.normAtrous4(self.convAtrous4(xC))))))

        xAtrous1 = nonlinearity(self.norm1x1_Atrous(self.conv1x1_Atrous(torch.cat((xAtrous1, xAtrous2, xAtrous3, xAtrous4), dim=1))))

        ######## POOLING ########
        xPool1 = nonlinearity(self.normLearnedPool1(self.convLearnedPool1(xC)))
        xPool2 = nonlinearity(self.normLearnedPool2(self.convLearnedPool2(xC)))
        xPool3 = nonlinearity(self.normLearnedPool3(self.convLearnedPool3(xC)))
        xPool4 = nonlinearity(self.normLearnedPool4(self.convLearnedPool4(xC)))

        xPool1 = F.upsample(xPool1, size=(h, w), mode='bilinear')
        xPool2 = F.upsample(xPool2, size=(h, w), mode='bilinear')
        xPool3 = F.upsample(xPool3, size=(h, w), mode='bilinear')
        xPool4 = F.upsample(xPool4, size=(h, w), mode='bilinear')

        xPool1 = nonlinearity(self.norm1x1_Pool(self.conv1x1_Pool(torch.cat((xPool1, xPool2, xPool3, xPool4), dim=1))))
        #########################

        x6 = nonlinearity(self.norm1x1_Both(self.conv1x1_Both(torch.cat((xAtrous1, xPool1), dim=1)))) + x6

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class CustomContext2(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomContext2, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = down(512, 1024, modelDim)

        # self.dacblock = DACblock(1024, modelDim=2, withAtrous=True)
        # self.rmpblock = RMPblock(1024, modelDim=2)

        ftAmount = 256

        self.conv1x1_Down = nn.Conv2d(4*ftAmount, ftAmount, kernel_size=1)
        self.norm1x1_Down = nn.BatchNorm2d(ftAmount)

        self.convAtrous1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=1, padding=1)
        self.normAtrous1 = nn.BatchNorm2d(ftAmount)
        self.convAtrous1_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=1, padding=1)
        self.normAtrous1_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=2, padding=2)
        self.normAtrous2 = nn.BatchNorm2d(ftAmount)
        self.convAtrous2_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=2, padding=2)
        self.normAtrous2_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=3, padding=3)
        self.normAtrous3 = nn.BatchNorm2d(ftAmount)
        self.convAtrous3_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=3, padding=3)
        self.normAtrous3_2 = nn.BatchNorm2d(ftAmount//4)

        self.convAtrous4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, dilation=4, padding=4)
        self.normAtrous4 = nn.BatchNorm2d(ftAmount)
        self.convAtrous4_2 = nn.Conv2d(ftAmount, ftAmount//4, kernel_size=3, dilation=4, padding=4)
        self.normAtrous4_2 = nn.BatchNorm2d(ftAmount//4)


        self.convLearnedPool1 = nn.Conv2d(ftAmount, ftAmount, kernel_size=2, stride=2, groups=ftAmount)
        self.normLearnedPool1 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool2 = nn.Conv2d(ftAmount, ftAmount, kernel_size=3, stride=3, groups=ftAmount)
        self.normLearnedPool2 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool3 = nn.Conv2d(ftAmount, ftAmount, kernel_size=5, stride=5, groups=ftAmount)
        self.normLearnedPool3 = nn.BatchNorm2d(ftAmount)

        self.convLearnedPool4 = nn.Conv2d(ftAmount, ftAmount, kernel_size=7, stride=7, groups=ftAmount)
        self.normLearnedPool4 = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Pool = nn.Conv2d(ftAmount*4, ftAmount, kernel_size=1)
        self.norm1x1_Pool = nn.BatchNorm2d(ftAmount)

        self.conv1x1_Both = nn.Conv2d(ftAmount*2, ftAmount*4, kernel_size=1)
        self.norm1x1_Both = nn.BatchNorm2d(ftAmount*4)


        # self.up0 = up(1024+4, 512, 512, modelDim, upsampling=False)
        self.up0 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False)
        self.outc = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        # x6 = self.dacblock(x6)
        # x6 = self.rmpblock(x6)

        in_channels, h, w = x6.size(1), x6.size(2), x6.size(3)

        xC = nonlinearity(self.norm1x1_Down(self.conv1x1_Down(x6)))

        ###### ATROUS CONV ######
        xAtrous1 = nonlinearity(self.normAtrous1_2(self.convAtrous1_2(nonlinearity(self.normAtrous1(self.convAtrous1(xC))))))
        xAtrous2 = nonlinearity(self.normAtrous2_2(self.convAtrous2_2(nonlinearity(self.normAtrous2(self.convAtrous2(xC))))))
        xAtrous3 = nonlinearity(self.normAtrous3_2(self.convAtrous3_2(nonlinearity(self.normAtrous3(self.convAtrous3(xC))))))
        xAtrous4 = nonlinearity(self.normAtrous4_2(self.convAtrous4_2(nonlinearity(self.normAtrous4(self.convAtrous4(xC))))))


        ######## POOLING ########
        xPool1 = nonlinearity(self.normLearnedPool1(self.convLearnedPool1(xC)))
        xPool2 = nonlinearity(self.normLearnedPool2(self.convLearnedPool2(xC)))
        xPool3 = nonlinearity(self.normLearnedPool3(self.convLearnedPool3(xC)))
        xPool4 = nonlinearity(self.normLearnedPool4(self.convLearnedPool4(xC)))

        xPool1 = F.upsample(xPool1, size=(h, w), mode='bilinear')
        xPool2 = F.upsample(xPool2, size=(h, w), mode='bilinear')
        xPool3 = F.upsample(xPool3, size=(h, w), mode='bilinear')
        xPool4 = F.upsample(xPool4, size=(h, w), mode='bilinear')

        xPool1 = nonlinearity(self.norm1x1_Pool(self.conv1x1_Pool(torch.cat((xPool1, xPool2, xPool3, xPool4), dim=1))))
        #########################

        x6 = nonlinearity(self.norm1x1_Both(self.conv1x1_Both(torch.cat((xAtrous1, xAtrous2, xAtrous3, xAtrous4, xPool1), dim=1)))) + x6


        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class CustomContextDACRMP(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CustomContextDACRMP, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 32, modelDim)
        # self.down1 = downContext(32, 64, modelDim)
        # self.down2 = downContext(64, 128, modelDim)
        # self.down3 = downContext(128, 256, modelDim)
        # self.down4 = downContext(256, 512, modelDim)
        self.down1 = down(32, 64, modelDim)
        self.down2 = down(64, 128, modelDim)
        self.down3 = down(128, 256, modelDim)
        self.down4 = down(256, 512, modelDim)
        self.down5 = downPadded(512, 1024, modelDim)

        self.dacblock = DACblock(1024, modelDim=2, withAtrous=True)
        self.rmpblock = RMPblock(1024, modelDim=2)


        # self.up1 = upContext(516, 256, 256, modelDim, upsampling=False)
        # self.up2 = upContext(256, 128, 128, modelDim, upsampling=False)
        # self.up3 = upContext(128, 64, 64, modelDim, upsampling=False)
        # self.up4 = upContext(64, 32, 32, modelDim, upsampling=False)
        self.up0 = upPadded(1028, 512, 512, modelDim, upsampling=False)
        self.up1 = up(512, 256, 256, modelDim, upsampling=False)
        self.up2 = up(256, 128, 128, modelDim, upsampling=False)
        self.up3 = up(128, 64, 64, modelDim, upsampling=False)
        self.up4 = up(64, 32, 32, modelDim, upsampling=False)
        self.outc = outconv(32, output_ch, modelDim)


        self.apply(init_weights_xavier_uniform)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x6 = self.dacblock(x6)
        x6 = self.rmpblock(x6)

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class downContext(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(downContext, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_res_block(in_ch, out_ch, modelDim)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class upContext(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False):
        super(upContext, self).__init__()
        self.modelDim = modelDim
        if upsampling:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)

        self.conv = conv_res_block(in_ch_1 + in_ch_2, out_ch, modelDim)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        # if self.modelDim == 2: #2D
        #     x1 = F.pad(x1, (0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        # else: #3D
        #     x1 = F.pad(x1, (0, x2.size()[4] - x1.size()[4], 0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        x = torch.cat([x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3], x1], dim=1)
        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class conv_res_block(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_res_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//4, kernel_size=1),
            nn.InstanceNorm2d(in_ch//4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch//4, in_ch//4, kernel_size=3),
            nn.InstanceNorm2d(in_ch//4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch // 4, out_ch // 4, kernel_size=1),
            nn.InstanceNorm2d(out_ch // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch // 4, kernel_size=3),
            nn.InstanceNorm2d(out_ch // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, kernel_size=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        # x1 = self.conv(x)
        # startIndexDim2 = (x.size()[2]-x1.size()[2])//2
        # startIndexDim3 = (x.size()[3]-x1.size()[3])//2
        # x = x[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3] + x1
        return x

class DACblockContext(nn.Module):
    def __init__(self, channel, modelDim, withAtrous=True):
        super(DACblockContext, self).__init__()
        if modelDim == 2:
            if withAtrous:
                self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1)
                self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3)
                self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5)
                self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1)
            else:
                self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(F.pad(x, (1, 1, 1, 1), mode='replicate')))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(F.pad(x, (3, 3, 3, 3), mode='replicate'))))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(F.pad(self.dilate1(F.pad(x, (1, 1, 1, 1), mode='replicate')), (3, 3, 3, 3), mode='replicate'))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(F.pad(self.dilate2(F.pad(self.dilate1(F.pad(x, (1, 1, 1, 1), mode='replicate')),(3, 3, 3, 3), mode='replicate')),(5, 5, 5, 5), mode='replicate'))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

####################################################################################################
# ----- Vanilla Unet 2D/3D - Pooling-Encoder + (Transposed/Upsampling)-Decoder + DoubleConvs ----- #
class UNetVanilla(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(UNetVanilla, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 64, modelDim)
        self.down1 = down(64, 128, modelDim)
        self.down2 = down(128, 256, modelDim)
        self.down3 = down(256, 512, modelDim)
        self.down4 = down(512, 1024, modelDim)
        self.up1 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up2 = up(512, 256, 256, modelDim, upsampling=False)
        self.up3 = up(256, 128, 128, modelDim, upsampling=False)
        self.up4 = up(128, 64, 64, modelDim, upsampling=False)
        self.outc = outconv(64, output_ch, modelDim)

        self.apply(init_weights_xavier_normal)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNetVanillaMod(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(UNetVanillaMod, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 64, modelDim)
        self.down1 = down(64, 128, modelDim)
        self.down2 = down(128, 256, modelDim)
        self.down3 = down(256, 512, modelDim)
        self.down4 = downNoPadding(512, 1024, modelDim)
        self.up1 = up(1024, 512, 512, modelDim, upsampling=False)
        self.up2 = up(512, 256, 256, modelDim, upsampling=False)
        self.up3 = up(256, 128, 128, modelDim, upsampling=False)
        self.up4 = up(128, 64, 64, modelDim, upsampling=False)
        self.outc = outconv(64, output_ch, modelDim)

        self.apply(init_weights_xavier_normal)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class StackedUNet(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(StackedUNet, self).__init__()
        assert modelDim == 2 or modelDim == 3, "Wrong unet-model dimension: " + str(modelDim)

        self.inc = initialconv(input_ch, 64, modelDim)
        self.down1 = down(64, 128, modelDim)
        self.down2 = down(128, 256, modelDim)
        self.down3 = down(256, 512, modelDim)
        self.down4 = down(512, 512, modelDim)
        self.up1 = up(512, 512, 256, modelDim, upsampling=False)
        self.up2 = up(256, 256, 128, modelDim, upsampling=False)
        self.up3 = up(128, 128, 64, modelDim, upsampling=False)
        self.up4 = up(64, 64, 64, modelDim, upsampling=False)
        self.outc = outconv(64, output_ch, modelDim)

        self.inc_2 = initialconv(output_ch, 32, modelDim)
        self.down1_2 = down(32, 64, modelDim)
        self.down2_2 = down(64, 128, modelDim)
        self.down3_2 = down(128, 256, modelDim)
        self.down4_2 = down(256, 256, modelDim)
        self.up1_2 = up(256, 256, 128, modelDim, upsampling=False)
        self.up2_2 = up(128, 128, 64, modelDim, upsampling=False)
        self.up3_2 = up(64, 64, 32, modelDim, upsampling=False)
        self.up4_2 = up(32, 32, 32, modelDim, upsampling=False)
        self.outc_2 = outconv(32, output_ch, modelDim)

        self.apply(init_weights_xavier_normal)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_Out = self.outc(x)

        x1 = self.inc_2(x_Out)
        x2 = self.down1_2(x1)
        x3 = self.down2_2(x2)
        x4 = self.down3_2(x3)
        x5 = self.down4_2(x4)
        x = self.up1_2(x5, x4)
        x = self.up2_2(x, x3)
        x = self.up3_2(x, x2)
        x = self.up4_2(x, x1)
        x = self.outc_2(x)

        return x_Out, x


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block, self).__init__()
        if modelDim == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
                # nn.Conv2d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                # nn.SELU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=2, dilation=2),
                # nn.Conv2d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(inplace=True)
                # nn.SELU(inplace=True)
            )
        elif modelDim == 3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv3d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv3d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            sys.exit('Wrong dimension '+str(modelDim)+' given!')

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_noPadding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block_noPadding, self).__init__()
        if modelDim == 2:
            self.conv = nn.Sequential(
                # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                # nn.SELU(inplace=True),
                # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm2d(out_ch),
                # nn.BatchNorm2d(out_ch),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(inplace=True)
                # nn.SELU(inplace=True)

                # nn.InstanceNorm2d(in_ch // 4),
                # nn.LeakyReLU(inplace=True),
                # # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv2d(in_ch // 4, in_ch // 4, kernel_size=3),
                # nn.InstanceNorm2d(in_ch // 4),
                # # nn.BatchNorm2d(out_ch),
                # # nn.ReLU(inplace=True),
                # nn.LeakyReLU(inplace=True),
                # # nn.SELU(inplace=True),
                # # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                # nn.Conv2d(in_ch // 4, in_ch // 4, kernel_size=3),
                # nn.InstanceNorm2d(in_ch // 4),
                # # nn.BatchNorm2d(out_ch),
                # # nn.ReLU(inplace=True)
                # nn.LeakyReLU(inplace=True),
                # # nn.SELU(inplace=True)
                # nn.Conv2d(in_ch // 4, out_ch, kernel_size=1),
                # nn.InstanceNorm2d(out_ch),
                # nn.LeakyReLU(inplace=True)
            )
        elif modelDim == 3:
            self.conv = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv3d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                # nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv3d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            sys.exit('Wrong dimension '+str(modelDim)+' given!')

    def forward(self, x):
        x = self.conv(x)
        return x

class conv5_block_noPadding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv5_block_noPadding, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=5),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            # nn.SELU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=5),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
            # nn.SELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_noPadding1x1(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block_noPadding1x1, self).__init__()
        if modelDim == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch//4, kernel_size=1),
                nn.InstanceNorm2d(in_ch//4),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_ch//4, in_ch//4, kernel_size=3),
                # nn.BatchNorm2d(out_ch),
                nn.InstanceNorm2d(in_ch//4),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_ch//4, in_ch//4, kernel_size=3),
                # nn.BatchNorm2d(out_ch),
                nn.InstanceNorm2d(in_ch//4),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(inplace=True),
                # nn.SELU(inplace=True)
                nn.Conv2d(in_ch//4, out_ch, kernel_size=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        elif modelDim == 3:
            self.conv = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv3d(in_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True),
                # nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.Conv3d(out_ch, out_ch, kernel_size=3),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            sys.exit('Wrong dimension '+str(modelDim)+' given!')

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_padding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(conv_block_padding, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.relu1(self.norm1(self.conv1(x)))
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = self.relu2(self.norm2(self.conv2(x)))
        return x



class initialconv(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(initialconv, self).__init__()
        self.conv = conv_block(in_ch, out_ch, modelDim)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(down, self).__init__()
        if modelDim == 2:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                # nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, groups=in_ch),
                conv_block(in_ch, out_ch, modelDim)
            )
        elif modelDim == 3:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                conv_block(in_ch, out_ch, modelDim)
            )
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class downNoPadding(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(downNoPadding, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block_noPadding(in_ch, out_ch, modelDim)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class downPadded(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(downPadded, self).__init__()
        if modelDim == 2:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                conv_block_padding(in_ch, out_ch, modelDim)
            )
        elif modelDim == 3:
            self.max_pool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                conv_block_padding(in_ch, out_ch, modelDim)
            )
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False, conv5=False):
        super(up, self).__init__()
        self.modelDim = modelDim
        if modelDim == 2:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                # self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1//2, 2, padding=0, stride=2)
                self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        elif modelDim == 3:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                # self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1//2, 2, padding=0, stride=2)
                self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

        # self.conv = conv_block_noPadding1x1(in_ch_1//2 + in_ch_2, out_ch, modelDim)
        if conv5:
            # self.conv = conv5_block_noPadding(in_ch_1//2 + in_ch_2, out_ch, modelDim)
            self.conv = conv5_block_noPadding(in_ch_1 + in_ch_2, out_ch, modelDim)
        else:
            # self.conv = conv_block_noPadding(in_ch_1//2 + in_ch_2, out_ch, modelDim)
            self.conv = conv_block_noPadding(in_ch_1 + in_ch_2, out_ch, modelDim)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        # if self.modelDim == 2: #2D
        #     x1 = F.pad(x1, (0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        # else: #3D
        #     x1 = F.pad(x1, (0, x2.size()[4] - x1.size()[4], 0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        x = torch.cat([x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3], x1], dim=1)
        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class upSparse(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False):
        super(upSparse, self).__init__()

        self.conv1 = nn.Conv2d(in_ch_1, in_ch_1 // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_ch_1 // 4)

        self.deconv2 = nn.ConvTranspose2d(in_ch_1 // 4, in_ch_1 // 4, 4, stride=2, padding=3)
        self.norm2 = nn.BatchNorm2d(in_ch_1 // 4)

        self.conv3 = nn.Conv2d(in_ch_1 // 4, in_ch_2, 1)
        self.norm3 = nn.BatchNorm2d(in_ch_2)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = nonlinearity(self.norm3(self.conv3(nonlinearity(self.norm2(self.deconv2(nonlinearity(self.norm1(self.conv1(x1)))))))))
        startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        x1 = x1 + x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3]

        return x1

class upPadded(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, modelDim, upsampling=False):
        super(upPadded, self).__init__()
        self.modelDim = modelDim
        if modelDim == 2:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        elif modelDim == 3:
            if upsampling:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose3d(in_ch_1, in_ch_1, 2, padding=0, stride=2)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

        self.conv = conv_block(in_ch_1 + in_ch_2, out_ch, modelDim)

    def forward(self, x1, x2): #x2 provides equal/decreased by 1 axis sizes
        x1 = self.up(x1)
        if self.modelDim == 2: #2D
            x1 = F.pad(x1, (0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        else: #3D
            x1 = F.pad(x1, (0, x2.size()[4] - x1.size()[4], 0, x2.size()[3] - x1.size()[3], 0, x2.size()[2] - x1.size()[2]), mode='replicate')
        # startIndexDim2 = (x2.size()[2]-x1.size()[2])//2
        # startIndexDim3 = (x2.size()[3]-x1.size()[3])//2
        # x = torch.cat([x2[:,:,startIndexDim2:x1.size()[2]+startIndexDim2, startIndexDim3:x1.size()[3]+startIndexDim3], x1], dim=1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, modelDim):
        super(outconv, self).__init__()
        if modelDim == 2:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        elif modelDim == 3:
            self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        x = self.conv(x)
        return x



class DACblock(nn.Module):
    def __init__(self, channel, modelDim, withAtrous=True):
        super(DACblock, self).__init__()
        if modelDim == 2:
            if withAtrous:
                self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
                self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
                self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
            else:
                self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        elif modelDim == 3:
            if withAtrous:
                self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=3, padding=3)
                self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=5, padding=5)
                self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
            else:
                self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
                self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class RMPblock(nn.Module): #RMP: (Residual Multi-Kernel Pooling)
    def __init__(self, in_channels, modelDim):
        super(RMPblock, self).__init__()
        self.modelDim = modelDim

        if modelDim == 2:
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
            self.pool3 = nn.MaxPool2d(kernel_size=5, stride=5)
            self.pool4 = nn.MaxPool2d(kernel_size=6, stride=6)

            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

        elif modelDim == 3:
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)
            self.pool3 = nn.MaxPool3d(kernel_size=5, stride=5)
            self.pool4 = nn.MaxPool3d(kernel_size=6, stride=6)

            self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        if self.modelDim == 2: #2D
            in_channels, h, w = x.size(1), x.size(2), x.size(3)
            layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
            layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
            layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
            layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        else: #3D
            in_channels, h, w, d = x.size(1), x.size(2), x.size(3), x.size(4)
            layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w, d), mode='trilinear')
            layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w, d), mode='trilinear')
            layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w, d), mode='trilinear')
            layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w, d), mode='trilinear')

        out = torch.cat([layer1, layer2, layer3, layer4, x], 1)
        return out


class Inception_resnet_v2_block(nn.Module):
    def __init__(self, channel, modelDim):
        super(Inception_resnet_v2_block, self).__init__()
        if modelDim == 2:
            self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
            self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
            self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        elif modelDim == 3:
            self.dilate1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
            self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
            self.conv1x1 = nn.Conv3d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out
        return out


class Inception_block(nn.Module):
    def __init__(self, channel, modelDim):
        super(Inception_block, self).__init__()
        if modelDim == 2:
            self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
            self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
            self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        elif modelDim == 3:
            self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
            self.conv3x3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
            self.conv5x5 = nn.Conv3d(channel, channel, kernel_size=5, dilation=1, padding=2)
            self.pooling = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        else:
            sys.exit('Wrong dimension ' + str(modelDim) + ' given!')

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out



# Care: Spatial output size is spatial input size * 2
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, modelDim):
        super(DecoderBlock, self).__init__()
        if modelDim == 2: #2D
            self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
            self.norm1 = nn.BatchNorm2d(in_channels // 4)
            # self.norm1 = nn.InstanceNorm2d(in_channels // 4)

            self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
            self.norm2 = nn.BatchNorm2d(in_channels // 4)
            # self.norm2 = nn.InstanceNorm2d(in_channels // 4)

            self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
            self.norm3 = nn.BatchNorm2d(n_filters)
            # self.norm3 = nn.InstanceNorm2d(n_filters)
        else: #3D
            self.conv1 = nn.Conv3d(in_channels, in_channels // 4, 1)
            self.norm1 = nn.BatchNorm3d(in_channels // 4)

            self.deconv2 = nn.ConvTranspose3d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
            self.norm2 = nn.BatchNorm3d(in_channels // 4)

            self.conv3 = nn.Conv3d(in_channels // 4, n_filters, 1)
            self.norm3 = nn.BatchNorm3d(n_filters)

    def forward(self, x):
        x = nonlinearity(self.norm1(self.conv1(x)))
        x = nonlinearity(self.norm2(self.deconv2(x)))
        x = nonlinearity(self.norm3(self.conv3(x)))
        return x


class CE_Net_2D(nn.Module):
    def __init__(self, output_ch=1, withAtrous=True):
        super(CE_Net_2D, self).__init__()

        filters = [64, 128, 256, 512]

        self.dacblock = DACblock(512, modelDim=2, withAtrous=withAtrous)
        self.rmpblock = RMPblock(512, modelDim=2)

        self.decoder4 = DecoderBlock(512+4, filters[2], modelDim=2)
        self.decoder3 = DecoderBlock(filters[2], filters[1], modelDim=2)
        self.decoder2 = DecoderBlock(filters[1], filters[0], modelDim=2)
        self.decoder1 = DecoderBlock(filters[0], filters[0], modelDim=2)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2d(32, output_ch, 3, padding=1)

        # filters = [64, 256, 512, 1024, 2048]
        #
        # self.dacblock = DACblock(2048, modelDim=2, withAtrous=withAtrous)
        # self.rmpblock = RMPblock(2048, modelDim=2)
        #
        # self.decoder4 = DecoderBlock(2048+4, filters[3], modelDim=2)
        # self.decoder3 = DecoderBlock(filters[3], filters[2], modelDim=2)
        # self.decoder2 = DecoderBlock(filters[2], filters[1], modelDim=2)
        # self.decoder1 = DecoderBlock(filters[1], filters[0], modelDim=2)
        #
        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalconv3 = nn.Conv2d(32, output_ch, 3, padding=1)


        self.apply(init_weights_xavier_normal)

        resnet = models.resnet34(pretrained=True)
        # resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1 # 3 x (conv-bn-relu-conv-bn)-blocks
        self.encoder2 = resnet.layer2 # 4 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder3 = resnet.layer3 # 6 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder4 = resnet.layer4 # 3 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        # self.apply(init_weights_xavier_normal)

    def forward(self, x0):
        # Encoder
        x = self.firstconv(x0)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0 = self.firstmaxpool(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dacblock(e4)
        e4 = self.rmpblock(e4)

        # Decoder
        d4 = self.decoder4(e4)
        d4 = F.pad(d4, (0, e3.size()[3] - d4.size()[3], 0, e3.size()[2] - d4.size()[2]), mode='replicate')
        d4 = d4 + e3
        d3 = self.decoder3(d4)
        d3 = F.pad(d3, (0, e2.size()[3] - d3.size()[3], 0, e2.size()[2] - d3.size()[2]), mode='replicate')
        d3 = d3 + e2
        d2 = self.decoder2(d3)
        d2 = F.pad(d2, (0, e1.size()[3] - d2.size()[3], 0, e1.size()[2] - d2.size()[2]), mode='replicate')
        d2 = d2 + e1
        d1 = self.decoder1(d2)
        d1 = F.pad(d1, (0, x.size()[3] - d1.size()[3], 0, x.size()[2] - d1.size()[2]), mode='replicate')
        d1 = d1 + x

        out = nonlinearity(self.finaldeconv1(d1))
        out = F.pad(out, (0, x0.size()[3] - out.size()[3], 0, x0.size()[2] - out.size()[2]), mode='replicate')
        out = nonlinearity(self.finalconv2(out))
        out = self.finalconv3(out)

        return out


class CE_Net_Inception_Variants_2D(nn.Module):
    def __init__(self, output_ch=1, inceptionBlock=True):
        super(CE_Net_Inception_Variants_2D, self).__init__()

        filters = [64, 128, 256, 512]

        if inceptionBlock:
            self.dblock = Inception_block(512, modelDim = 2)
        else:
            self.dblock = Inception_resnet_v2_block(512, modelDim = 2)
        # self.rmpblock = RMPblock(512, modelDim=2)

        self.decoder4 = DecoderBlock(512+4, filters[2], modelDim=2)
        self.decoder3 = DecoderBlock(filters[2], filters[1], modelDim=2)
        self.decoder2 = DecoderBlock(filters[1], filters[0], modelDim=2)
        self.decoder1 = DecoderBlock(filters[0], filters[0], modelDim=2)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2d(32, output_ch, 3, padding=1)

        self.apply(init_weights_xavier_normal)

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1 # 3 x (conv-bn-relu-conv-bn)-blocks
        self.encoder2 = resnet.layer2 # 4 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder3 = resnet.layer3 # 6 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block
        self.encoder4 = resnet.layer4 # 3 x (conv-bn-relu-conv-bn)-blocks including 2-strided conv + bn after first block


    def forward(self, x0):
        # Encoder
        x = self.firstconv(x0)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0 = self.firstmaxpool(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        # e4 = self.rmpblock(e4)

        # Decoder
        d4 = self.decoder4(e4)
        d4 = F.pad(d4, (0, e3.size()[3] - d4.size()[3], 0, e3.size()[2] - d4.size()[2]), mode='replicate')
        d4 = d4 + e3
        d3 = self.decoder3(d4)
        d3 = F.pad(d3, (0, e2.size()[3] - d3.size()[3], 0, e2.size()[2] - d3.size()[2]), mode='replicate')
        d3 = d3 + e2
        d2 = self.decoder2(d3)
        d2 = F.pad(d2, (0, e1.size()[3] - d2.size()[3], 0, e1.size()[2] - d2.size()[2]), mode='replicate')
        d2 = d2 + e1
        d1 = self.decoder1(d2)
        d1 = F.pad(d1, (0, x.size()[3] - d1.size()[3], 0, x.size()[2] - d1.size()[2]), mode='replicate')
        d1 = d1 + x

        out = nonlinearity(self.finaldeconv1(d1))
        out = F.pad(out, (0, x0.size()[3] - out.size()[3], 0, x0.size()[2] - out.size()[2]), mode='replicate')
        out = nonlinearity(self.finalconv2(out))
        out = self.finalconv3(out)

        return out



##############################################################################################################
# ----- Classic FCN - Encoder: Double conv + Max-Pooling -> Decoder: Upsampling ftMaps from each level ----- #
class FCNClassical(nn.Module):
    def __init__(self, input_ch, output_ch, modelDim):
        super(FCNClassical, self).__init__()
        self.modelDim = modelDim
        if modelDim == 2: #2D
            # contraction:
            self.b1_cv1 = nn.Conv2d(input_ch, 32, 3, padding=1)
            self.b1_cv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.b1_mp = nn.MaxPool2d(3, stride=2, padding=1)

            self.b2_cv1 = nn.Conv2d(32, 64, 3, padding=1)
            self.b2_cv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.b2_mp = nn.MaxPool2d(3, stride=2, padding=1)

            self.b3_cv1 = nn.Conv2d(64, 128, 3, padding=1)
            self.b3_cv2 = nn.Conv2d(128, 128, 3, padding=1)
            self.b3_mp = nn.MaxPool2d(3, stride=2, padding=1)

            self.b4_cv1 = nn.Conv2d(128, 192, 3, padding=1)
            self.b4_cv2 = nn.Conv2d(192, 192, 3, padding=1)
            self.b4_mp = nn.MaxPool2d(3, stride=2, padding=1)

            self.b5_cv1 = nn.Conv2d(192, 256, 3, padding=1)
            self.b5_cv2 = nn.Conv2d(256, 256, 3, padding=1)
            self.b5_mp = nn.MaxPool2d(3, stride=2, padding=1)

            self.b6_cv1 = nn.Conv2d(256, 384, 3, padding=1)
            self.b6_cv2 = nn.Conv2d(384, 384, 3, padding=1)
            self.b6_mp = nn.MaxPool2d(3, stride=2, padding=1)

            # reconstruction
            self.rec_cv1 = nn.Conv2d(1056, 64, 3, padding=1)
            self.rec_cv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.rec_cv3 = nn.Conv2d(64, output_ch, 3, padding=1)

            self.dropOut = torch.nn.Dropout2d(p = 0.05)
        else: #3D
            # contraction:
            self.b1_cv1 = nn.Conv3d(input_ch, 32, 3, padding=1)
            self.b1_cv2 = nn.Conv3d(32, 32, 3, padding=1)
            self.b1_mp = nn.MaxPool3d(3, stride=2, padding=1)

            self.b2_cv1 = nn.Conv3d(32, 64, 3, padding=1)
            self.b2_cv2 = nn.Conv3d(64, 64, 3, padding=1)
            self.b2_mp = nn.MaxPool3d(3, stride=2, padding=1)

            self.b3_cv1 = nn.Conv3d(64, 128, 3, padding=1)
            self.b3_cv2 = nn.Conv3d(128, 128, 3, padding=1)
            self.b3_mp = nn.MaxPool3d(3, stride=2, padding=1)

            self.b4_cv1 = nn.Conv3d(128, 192, 3, padding=1)
            self.b4_cv2 = nn.Conv3d(192, 192, 3, padding=1)
            self.b4_mp = nn.MaxPool3d(3, stride=2, padding=1)

            self.b5_cv1 = nn.Conv3d(192, 256, 3, padding=1)
            self.b5_cv2 = nn.Conv3d(256, 256, 3, padding=1)
            self.b5_mp = nn.MaxPool3d(3, stride=2, padding=1)

            self.b6_cv1 = nn.Conv3d(256, 384, 3, padding=1)
            self.b6_cv2 = nn.Conv3d(384, 384, 3, padding=1)
            self.b6_mp = nn.MaxPool3d(3, stride=2, padding=1)

            # reconstruction
            self.rec_cv1 = nn.Conv3d(1056, 64, 3, padding=1)
            self.rec_cv2 = nn.Conv3d(64, 64, 3, padding=1)
            self.rec_cv3 = nn.Conv3d(64, output_ch, 3, padding=1)

            self.dropOut = torch.nn.Dropout3d(p=0.05)

        self.apply(init_weights_xavier_normal)


    def forward(self, x):
        res2 = self.b1_mp(F.softplus(self.b1_cv2(self.b1_cv1(x))))
        res4 = self.dropOut(self.b2_mp(F.softplus(self.b2_cv2(self.b2_cv1(res2)))))
        res8 = self.dropOut(self.b3_mp(F.softplus(self.b3_cv2(self.b3_cv1(res4)))))
        res16 = self.dropOut(self.b4_mp(F.softplus(self.b4_cv2(self.b4_cv1(res8)))))
        res32 = self.dropOut(self.b5_mp(F.softplus(self.b5_cv2(self.b5_cv1(res16)))))
        res64 = self.dropOut(self.b6_mp(F.softplus(self.b6_cv2(self.b6_cv1(res32)))))

        spatialSizeRes2 = res2.size()[2:]
        y = torch.cat((
            res2,
            F.upsample(res4, size=spatialSizeRes2),
            F.upsample(res8, size=spatialSizeRes2),
            F.upsample(res16, size=spatialSizeRes2),
            F.upsample(res32, size=spatialSizeRes2),
            F.upsample(res64, size=spatialSizeRes2)
        ), dim=1)

        if self.modelDim == 2: #2D
            y = F.upsample(F.softplus(self.rec_cv1(y)), mode='bilinear', size=x.size()[2:])
        else: #3D
            y = F.upsample(F.softplus(self.rec_cv1(y)), mode='trilinear', size=x.size()[2:])
        y = self.rec_cv3(F.softplus(self.rec_cv2(y)))
        return y



##############################################################################################################
# ----- 3D Custom Unet - Encoder: double conv + Max-Pooling, Decoder: single conv + Trilinear sampling ----- #
class Unet_OwnVariant_3D(nn.Module):

    def __init__(self, input_ch, output_ch):

        super(Unet_OwnVariant_3D, self).__init__()

        # Classic all-conv.-Unet [880k parameters]
        self.conv0 = nn.Conv3d(input_ch, 5, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(5)

        self.conv1 = nn.Conv3d(5, 14, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(14)
        self.conv11 = nn.Conv3d(14, 14, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(14)

        self.conv2 = nn.Conv3d(14, 28, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(28)
        self.conv22 = nn.Conv3d(28, 28, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(28)

        self.conv3 = nn.Conv3d(28, 42, 3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm3d(42)
        self.conv33 = nn.Conv3d(42, 42, 3, padding=1)
        self.batch33 = nn.BatchNorm3d(42)

        self.conv4 = nn.Conv3d(42, 56, 3, stride=2, padding=1)
        self.batch4 = nn.BatchNorm3d(56)
        self.conv44 = nn.Conv3d(56, 56, 3, padding=1)
        self.batch44 = nn.BatchNorm3d(56)

        self.conv5 = nn.Conv3d(56, 70, 3, stride=2, padding=1)
        self.batch5 = nn.BatchNorm3d(70)
        self.conv55 = nn.Conv3d(70, 70, 3, padding=1)
        self.batch55 = nn.BatchNorm3d(70)

        self.conv6dU = nn.Conv3d(126, 56, 3, padding=1)
        self.batch6dU = nn.BatchNorm3d(56)

        self.conv6cU = nn.Conv3d(98, 42, 3, padding=1)
        self.batch6cU = nn.BatchNorm3d(42)

        self.conv6bU = nn.Conv3d(70, 28, 3, padding=1)
        self.batch6bU = nn.BatchNorm3d(28)

        self.conv6U = nn.Conv3d(42, 14, 3, padding=1)
        self.batch6U = nn.BatchNorm3d(14)

        self.conv7U = nn.Conv3d(19, output_ch, 3, padding=1)
        self.batch7U = nn.BatchNorm3d(output_ch)
        self.conv77U = nn.Conv3d(output_ch, output_ch, 3, padding=1)

        self.apply(init_weights_xavier_normal)


    def forward(self, inputImg):
        x1 = F.dropout3d(F.leaky_relu(self.batch0(self.conv0(inputImg)), 0.1))

        x = F.dropout3d(F.leaky_relu(self.batch1(self.conv1(x1)),0.1))
        x2 = F.dropout3d(F.leaky_relu(self.batch11(self.conv11(x)),0.1))

        x = F.dropout3d(F.leaky_relu(self.batch2(self.conv2(x2)),0.1))
        x3 = F.dropout3d(F.leaky_relu(self.batch22(self.conv22(x)),0.1))

        x = F.dropout3d(F.leaky_relu(self.batch3(self.conv3(x3)),0.1))
        x4 = F.dropout3d(F.leaky_relu(self.batch33(self.conv33(x)),0.1))

        x = F.dropout3d(F.leaky_relu(self.batch4(self.conv4(x4)),0.1))
        x5 = F.dropout3d(F.leaky_relu(self.batch44(self.conv44(x)),0.1))

        x = F.dropout3d(F.leaky_relu(self.batch5(self.conv5(x5)),0.1))
        x = F.dropout3d(F.leaky_relu(self.batch55(self.conv55(x)),0.1))

        x = F.upsample(x, size=x5.size()[2:], mode='trilinear')
        x = F.dropout3d(F.leaky_relu(self.batch6dU(self.conv6dU(torch.cat((x,x5),1))),0.1))

        x = F.upsample(x, size=x4.size()[2:], mode='trilinear')
        x = F.dropout3d(F.leaky_relu(self.batch6cU(self.conv6cU(torch.cat((x,x4),1))),0.1))

        x = F.upsample(x, size=x3.size()[2:], mode='trilinear')
        x = F.dropout3d(F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x3),1))),0.1))

        x = F.upsample(x, size=x2.size()[2:], mode='trilinear')
        x = F.dropout3d(F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x2),1))),0.1))

        x = F.upsample(x, size=x1.size()[2:], mode='trilinear')
        x = F.dropout3d(F.leaky_relu(self.batch7U(self.conv7U(torch.cat((x,x1),1))),0.1))
        x = self.conv77U(x)

        return x



########## small bottleneck module ##########
class BottleneckBlock(nn.Module):
    def __init__(self, channels, compress=8, modelDim=2):
        K = channels // compress
        # assert (K * compress == channels)
        super(BottleneckBlock, self).__init__()

        self.activation = nn.SELU(inplace=True)
        if modelDim == 2: #2D
            self.cv1 = nn.Conv2d(channels, K, 1)
            self.cv2 = nn.Conv2d(K, K, 3, padding=1)
            self.cv3 = nn.Conv2d(K, channels, 1)
        else: #3D
            self.cv1 = nn.Conv3d(channels, K, 1)
            self.cv2 = nn.Conv3d(K, K, 3, padding=1)
            self.cv3 = nn.Conv3d(K, channels, 1)

    def forward(self, x):
        y = self.activation(self.cv1(x))
        y = self.activation(self.cv2(y))
        y = self.activation(self.cv3(y))

        return torch.add(x, y)




