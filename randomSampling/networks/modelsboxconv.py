#ºimport numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable




class ENet(nn.ModuleList):
    def __init__(self, n_bands=3,n_classes=19):
        initfilter = 64
        super().__init__([
            #Downsampler(n_bands, 16),
            nn.Conv2d(n_bands, initfilter//2, 1,1),
            Bottleneck(initfilter//2, initfilter, 0.01, downsample=True),

            Bottleneck(initfilter, initfilter, 0.01),
            Bottleneck(initfilter, initfilter, 0.01),
            Bottleneck(initfilter, initfilter, 0.01),
            Bottleneck(initfilter, initfilter, 0.01),

            Bottleneck(initfilter, initfilter*2, 0.1, downsample=True),

            Bottleneck(initfilter*2, initfilter*2, 0.1),
            Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=2),
            Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=4),
            Bottleneck(initfilter*2, initfilter*4, 0.1, downsample=True),
            Bottleneck(initfilter*4, initfilter*4, 0.1, dilation=8),
            Bottleneck(initfilter*4, initfilter*4, 0.1, asymmetric_ksize=3),
            Bottleneck(initfilter*4, initfilter*4, 0.1, dilation=16),

            Bottleneck(initfilter*4, initfilter*8, 0.1, downsample=True),
            Bottleneck(initfilter*8, initfilter*8, 0.1, dilation=2),
            Bottleneck(initfilter*8, initfilter*8, 0.1, asymmetric_ksize=3),
            Bottleneck(initfilter*8, initfilter*8, 0.1, dilation=4),
            Bottleneck(initfilter*8, initfilter*8, 0.1),
            Bottleneck(initfilter*8, initfilter*8, 0.1, dilation=8),
            Bottleneck(initfilter*8, initfilter*8, 0.1, asymmetric_ksize=3),
            Bottleneck(initfilter*8, initfilter*8, 0.1, dilation=16),

            Upsampler(initfilter*8, initfilter*4),
            Upsampler(initfilter*4, initfilter*2),
            Upsampler(initfilter*2, initfilter),

            Bottleneck(initfilter, initfilter, 0.1),
            Bottleneck(initfilter, initfilter, 0.1),

            Upsampler(initfilter, initfilter//2),

            Bottleneck(initfilter//2, initfilter//2, 0.1),

            #nn.ConvTranspose2d(16, n_classes, (2,2), (2,2))])
            #nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(initfilter//2, n_classes, 1)
            ])

#class ENet(nn.ModuleList):
    #def __init__(self, n_bands=3,n_classes=19):
        #initfilter = 64
        #super().__init__([
            ##Downsampler(n_bands, 16),
            #nn.Conv2d(n_bands, initfilter//2, 1,1),
            #Bottleneck(initfilter//2, initfilter, 0.01, downsample=True),

            #Bottleneck(initfilter, initfilter, 0.01),
            #Bottleneck(initfilter, initfilter, 0.01),
            #Bottleneck(initfilter, initfilter, 0.01),
            #Bottleneck(initfilter, initfilter, 0.01),

            #Bottleneck(initfilter, initfilter*2, 0.1, downsample=True),

            #Bottleneck(initfilter*2, initfilter*2, 0.1),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=2),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=4),
            #Bottleneck(initfilter*2, initfilter*2, 0.1),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=8),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=16),

            #Bottleneck(initfilter*2, initfilter*2, 0.1),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=2),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=4),
            #Bottleneck(initfilter*2, initfilter*2, 0.1),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=8),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, dilation=16),

            #Upsampler(initfilter*2, initfilter),

            #Bottleneck(initfilter, initfilter, 0.1),
            #Bottleneck(initfilter, initfilter, 0.1),

            #Upsampler(initfilter, initfilter//2),

            #Bottleneck(initfilter//2, initfilter//2, 0.1),

            ##nn.ConvTranspose2d(16, n_classes, (2,2), (2,2))])
            ##nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            #nn.Conv2d(initfilter//2, n_classes, 1)
            #])

    def forward(self, x):
        max_indices_stack = []

        for module in self:
            if isinstance(module, Upsampler):
                x = module(x, max_indices_stack.pop())
            else:
                x = module(x)

            if type(x) is tuple: # then it was a downsampling bottleneck block
                x, max_indices = x
                max_indices_stack.append(max_indices)
        return x



class ENetPequena(nn.ModuleList):
    def __init__(self, n_bands=3,n_classes=19):
        initfilter = 64
        super().__init__([
            Downsampler(n_bands, initfilter),
            #nn.Conv2d(n_bands, initfilter//2, 1,1),
            Bottleneck(initfilter//2, initfilter, 0.01, downsample=True),

            Bottleneck(initfilter, initfilter, 0.1),
            Bottleneck(initfilter, initfilter, 0.1, dilation=2),
            Bottleneck(initfilter, initfilter, 0.1, asymmetric_ksize=3),
            Bottleneck(initfilter, initfilter, 0.1, dilation=8),
            Bottleneck(initfilter, initfilter, 0.1, asymmetric_ksize=3),
            Bottleneck(initfilter, initfilter, 0.1, dilation=4),

            Upsampler(initfilter, initfilter//2),

            Bottleneck(initfilter//2, initfilter//2, 0.1),

            #nn.ConvTranspose2d(16, n_classes, (2,2), (2,2))])
            #nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(initfilter//2, n_classes, 1)
            ])


    def forward(self, x):
        max_indices_stack = []

        for module in self:
            if isinstance(module, Upsampler):
                x = module(x, max_indices_stack.pop())
            else:
                x = module(x)

            if type(x) is tuple: # then it was a downsampling bottleneck block
                x, max_indices = x
                max_indices_stack.append(max_indices)
        return x
    
class BoxENet(ENet):
    def __init__(self,n_bands=3, n_classes=19, max_input_h=64, max_input_w=64):
        h, w = max_input_h, max_input_w # shorten names for convenience
        r = 1.5625 # reparametrization factor
        initfilter = 64
        nn.ModuleList.__init__(self, [
            #Downsampler(n_bands, 16),
            nn.Conv2d(n_bands, initfilter//2, 1,1),
            Bottleneck(initfilter//2, initfilter, 0.01, downsample=True),
            #Bottleneck(64, 64, 0.01, downsample=True),

            Bottleneck(initfilter, initfilter, 0.01),
            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),
            Bottleneck(initfilter, initfilter, 0.01),
            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),

            Bottleneck(initfilter, initfilter*2, 0.1, downsample=True),

            Bottleneck(initfilter*2, initfilter*2, 0.1),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            Bottleneck(initfilter*2, initfilter*4, 0.1, downsample=True),
            BottleneckBoxConv(initfilter*4, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            Bottleneck(initfilter*4, initfilter*4, 0.1, asymmetric_ksize=3),
            BottleneckBoxConv(initfilter*4, 4, h // 4, w // 4, 0.25, reparam_factor=r),

            Bottleneck(initfilter*4, initfilter*8, 0.1,downsample=True),
            BottleneckBoxConv(initfilter*8, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            Bottleneck(initfilter*8, initfilter*8, 0.1, asymmetric_ksize=3),
            BottleneckBoxConv(initfilter*8, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            Bottleneck(initfilter*8, initfilter*8, 0.1),
            BottleneckBoxConv(initfilter*8, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            Bottleneck(initfilter*8, initfilter*8, 0.1, asymmetric_ksize=3),
            BottleneckBoxConv(initfilter*8, 4, h // 4, w // 4, 0.25, reparam_factor=r),

            Upsampler(initfilter*8, initfilter*4),
            Upsampler(initfilter*4, initfilter*2),
            Upsampler(initfilter*2, initfilter),
            Bottleneck(initfilter, initfilter, 0.1),
            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.1, reparam_factor=r),

            Upsampler(initfilter, initfilter//2),

            BottleneckBoxConv(initfilter//2, 2, h // 2, w // 2, 0.1, reparam_factor=r),

            #nn.ConvTranspose2d(16, n_classes, (2,2), (2,2))])
            #nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(initfilter//2, n_classes, 1)
            ])

#class BoxENet(ENet):
    #def __init__(self,n_bands=3, n_classes=19, max_input_h=64, max_input_w=64):
        #h, w = max_input_h, max_input_w # shorten names for convenience
        #r = 1.5625 # reparametrization factor
        #initfilter = 64
        #nn.ModuleList.__init__(self, [
            ##Downsampler(n_bands, 16),
            #nn.Conv2d(n_bands, initfilter//2, 1,1),
            #Bottleneck(initfilter//2, initfilter, 0.01, downsample=True),
            ##Bottleneck(64, 64, 0.01, downsample=True),

            #Bottleneck(initfilter, initfilter, 0.01),
            #BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),
            #Bottleneck(initfilter, initfilter, 0.01),
            #BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),

            #Bottleneck(initfilter, initfilter*2, 0.1, downsample=True),

            #Bottleneck(initfilter*2, initfilter*2, 0.1),
            #BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            #BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            #Bottleneck(initfilter*2, initfilter*2, 0.1),
            #BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            #BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),

            #Bottleneck(initfilter*2, initfilter*2, 0.1),
            #BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            #BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            #Bottleneck(initfilter*2, initfilter*2, 0.1),
            #BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            #Bottleneck(initfilter*2, initfilter*2, 0.1, asymmetric_ksize=3),
            #BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.25, reparam_factor=r),

            #Upsampler(initfilter*2, initfilter),

            #Bottleneck(initfilter, initfilter, 0.1),
            #BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.1, reparam_factor=r),

            #Upsampler(initfilter, initfilter//2),

            #BottleneckBoxConv(initfilter//2, 2, h // 2, w // 2, 0.1, reparam_factor=r),

            ##nn.ConvTranspose2d(16, n_classes, (2,2), (2,2))])
            ##nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            #nn.Conv2d(initfilter//2, n_classes, 1)
            #])


class BoxOnlyENet(ENet):
    def __init__(self,n_bands=3, n_classes=19, max_input_h=64, max_input_w=64):
        h, w = max_input_h, max_input_w # shorten names for convenience
        r = 1.5625 # reparametrization factor
        initfilter = 64
        nn.ModuleList.__init__(self, [
            #Downsampler(n_bands, initfilter//2),
            #Bottleneck(16, 64, 0.01, downsample=True),
            nn.Conv2d(n_bands, initfilter//2, 1,1),
            
            Bottleneck(initfilter//2, initfilter, 0.01, downsample=True),

            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),

            Bottleneck(initfilter, initfilter*2, 0.01, downsample=True),

            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),

            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter*2, 4, h // 4, w // 4, 0.4, reparam_factor=r),

            Upsampler(initfilter*2, initfilter),

            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),

            Upsampler(initfilter, initfilter//2),

            BottleneckBoxConv(initfilter//2, 4, h // 2, w // 2, 0.05, reparam_factor=r),

            #nn.ConvTranspose2d(initfilter//2, n_classes, (2,2), (2,2))])
            #nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(initfilter//2, n_classes, 1),
   
            ])





class BoxENetPequena(ENet):
    def __init__(self,n_bands=3, n_classes=19, max_input_h=64, max_input_w=64):
        h, w = max_input_h, max_input_w # shorten names for convenience
        r = 1.5625 # reparametrization factor
        initfilter = 64
        nn.ModuleList.__init__(self, [
            #Downsampler(n_bands, 16),
            nn.Conv2d(n_bands, initfilter//2, 1,1),
          
            Bottleneck(initfilter//2, initfilter, 0.01, downsample=True),
            #Bottleneck(64, 64, 0.01, downsample=True),

            #Bottleneck(initfilter, initfilter, 0.01),
            #BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.15, reparam_factor=r),

            Bottleneck(initfilter, initfilter, 0.1),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            Bottleneck(initfilter, initfilter, 0.1, asymmetric_ksize=3),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.25, reparam_factor=r),
            Bottleneck(initfilter, initfilter, 0.1, asymmetric_ksize=3),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.25, reparam_factor=r),


            #Bottleneck(initfilter, initfilter, 0.1),
            #BottleneckBoxConv(initfilter, 4, h // 2, w // 2, 0.1, reparam_factor=r),

            Upsampler(initfilter, initfilter//2),

            BottleneckBoxConv(initfilter//2, 2, h // 2, w // 2, 0.1, reparam_factor=r),

            #nn.ConvTranspose2d(16, n_classes, (2,2), (2,2))])
            #nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(initfilter//2, n_classes, 1)
            ])


class BoxOnlyENetPequena(ENet):
    def __init__(self,n_bands=3, n_classes=19, max_input_h=64, max_input_w=64):
        h, w = max_input_h, max_input_w # shorten names for convenience
        r = 0.510 # reparametrization factor
        initfilter = 64
        nn.ModuleList.__init__(self, [
            #Downsampler(n_bands, 16),
            #Bottleneck(16, 64, 0.01, downsample=True),
            nn.Conv2d(n_bands, initfilter//2, 1,1),
            Bottleneck(initfilter//2, initfilter, 0.01, downsample=True),

            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.4, reparam_factor=r),
            BottleneckBoxConv(initfilter, 4, h // 4, w // 4, 0.4, reparam_factor=r),

            Upsampler(initfilter, initfilter//2),

            BottleneckBoxConv(initfilter//2, 4, h // 2, w // 2, 0.05, reparam_factor=r),

            #nn.ConvTranspose2d(16, n_classes, (2,2), (2,2))])
            #nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(initfilter//2, n_classes, 1)
            ])

class ENetMinus(ENet):
    def __init__(self,n_bands=3, n_classes=19, max_input_h=512, max_input_w=1024):
        h, w = max_input_h, max_input_w # shorten names for convenience
        r = 0.860 # reparametrization factor

        nn.ModuleList.__init__(self, [
            Downsampler(n_bands, 16),
            Bottleneck(16, 64, 0.01, downsample=True),

            Bottleneck(64, 64, 0.01),
            Bottleneck(64, 64, 0.01),

            Bottleneck(64, 128, 0.1, downsample=True),

            Bottleneck(128, 128, 0.1),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=3),
            Bottleneck(128, 128, 0.1),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=3),

            Bottleneck(128, 128, 0.1),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=3),
            Bottleneck(128, 128, 0.1),
            Bottleneck(128, 128, 0.1, asymmetric_ksize=3),

            Upsampler(128, 64),

            Bottleneck(64, 64, 0.1),

            Upsampler(64, 16),

            #nn.ConvTranspose2d(16, n_classes, (2,2), (2,2))])
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            nn.Conv2d(16, n_classes, 1)
            ])



class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        bt_channels = out_channels // 4

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1,1), bias=False),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(bt_channels, bt_channels, (3,3), 2, 1, 1),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.ReLU(True),

            nn.Conv2d(bt_channels, out_channels, (1,1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3))

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1,1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3))

    def forward(self, x, max_indices):
        x_skip_connection = self.skip_connection(x)
        x_skip_connection = F.max_unpool2d(x_skip_connection, max_indices, (2,2))
        return (x_skip_connection + self.main_branch(x)).relu_()


class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3,3), 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, 1e-3)
        self.prelu = nn.PReLU(out_channels)
        #self.relu = nn.ReLU(out_channels)
    def forward(self, x):
        x = torch.cat([F.max_pool2d(x, (2,2)), self.conv(x)], 1)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0, downsample=False,
        asymmetric_ksize=None, dilation=1, use_prelu=False):

        super().__init__()
        bt_channels = out_channels // 4
        self.downsample = downsample
        self.channels_to_pad = out_channels-in_channels

        input_stride = 2 if downsample else 1

        main_branch = [
            nn.Conv2d(in_channels, bt_channels, input_stride, input_stride, bias=False),
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.PReLU(bt_channels) if use_prelu else nn.ReLU(True)
        ]
       
        if asymmetric_ksize is None:
            main_branch += [
                nn.Conv2d(bt_channels, bt_channels, (3,3), 1, dilation, dilation, bias=False)
            ]
        else:
            assert type(asymmetric_ksize) is int
            ksize, padding = asymmetric_ksize, (asymmetric_ksize-1) // 2
            main_branch += [
                nn.Conv2d(bt_channels, bt_channels, (ksize,1), 1, (padding,0), bias=False),
                nn.Conv2d(bt_channels, bt_channels, (1,ksize), 1, (0,padding))
            ]
       
        main_branch += [
            nn.BatchNorm2d(bt_channels, 1e-3),
            nn.PReLU(bt_channels) if use_prelu else nn.ReLU(True),
            nn.Conv2d(bt_channels, out_channels, (1,1), bias=False),
            nn.BatchNorm2d(out_channels, 1e-3)
            #nn.Dropout2d(dropout_prob)
        ]

        self.main_branch = nn.Sequential(*main_branch)        
        self.output_activation = nn.PReLU(out_channels) if use_prelu else nn.ReLU(True)

    def forward(self, x):
        if self.downsample:
            x_skip_connection, max_indices = F.max_pool2d(x, (2,2), return_indices=True)
        else:
            x_skip_connection = x

        if self.channels_to_pad > 0:
            x_skip_connection = F.pad(x_skip_connection, (0,0, 0,0, 0,self.channels_to_pad))

        x = self.output_activation(x_skip_connection + self.main_branch(x))
        
        if self.downsample:
            return x, max_indices
        else:
            return x

from box_convolution import BoxConv2d

class BottleneckBoxConv(nn.Module):
    def __init__(self, in_channels, num_boxes, max_input_h, max_input_w,
        dropout_prob=0.0, reparam_factor=1.5625):

        super().__init__()
        assert in_channels % num_boxes == 0
        bt_channels = in_channels // num_boxes # bottleneck channels

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1,1), bias=False),
            nn.BatchNorm2d(bt_channels),
           
            
            # BEHOLD:
            BoxConv2d(
                bt_channels, num_boxes, max_input_h, max_input_w,
                reparametrization_factor=reparam_factor),

            nn.BatchNorm2d(in_channels,1e-3),
            nn.ReLU(True))
            #nn.Dropout2d(dropout_prob))

    def forward(self, x):
        return (x + self.main_branch(x)).relu_()





if __name__ == '__main__':
    network = SegNet(3,20)
    print(network)
