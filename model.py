import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial


##### XAVIER WEIGHT INITIALIZATION FOR NETWORK PARAMETER INITIALIZATION #####
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



nonlinearity = partial(F.relu, inplace=True)

####################################################################################################
# Custom represents our utilized and developed deep learning model. It is based on the U-Net architecture:
# ----- Custom Unet 2D - Pooling-Encoder + (Transposed/Upsampling)-Decoder + DoubleConvs ----- #
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

# This class represents the vanilla u-net by Ronneberger et al. that has been sparsely modified to run on 640x640 data and output 516x516
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

# This class represents the vanilla context-encoder by Gu et al. that has been sparsely modified to run on 640x640 data and output 516x516
# Code (like DAC, RMP modules, etc) mostly copied from his github repo: https://github.com/Guzaiwang/CE-Net
class CE_mod(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, modelDim=2):
        super(CE_mod, self).__init__()
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

