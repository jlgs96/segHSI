#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:56:02 2019

@author: aneesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .selayer import ChannelSELayer
from box_convolution import BoxConv2d


class unetConv2(nn.Module):
    '''
    U-Net encoder block with Squeeze and Excitation layer flag and
    a default kernel size of 3
    
    Parameters:
        in_size     -- number of input channels
        out_size    -- number of output channels
        is_batchnorm-- boolean flag to indicate batch-normalization usage 
        use_se      -- boolean flag to indicate if SE block is used
        act         -- flag to indicate activation between linear layers in SE 
                        (relu vs. prelu)
    ''' 
    def __init__(self, in_size, out_size, is_batchnorm, use_se = False, use_prelu = False, max_input_h=64, max_input_w=64, use_boxconv=False):
        super(unetConv2, self).__init__()
       
        if use_boxconv:
            if is_batchnorm:
                n_boxes =4
                
                #reparam_factor = 0.860
                reparam_factor = 0.860
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size//n_boxes, kernel_size = 1, stride = 1,padding = 0),nn.BatchNorm2d(out_size//n_boxes), nn.ReLU(),
                                           BoxConv2d(out_size//n_boxes,n_boxes,max_input_h,max_input_w,reparametrization_factor=reparam_factor), 
                                           nn.BatchNorm2d(out_size),#nn.Dropout(p = 0.5)
                                           
                )
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size//n_boxes, kernel_size = 1, stride = 1,padding = 0),nn.BatchNorm2d(out_size//n_boxes), nn.ReLU(),
                                           BoxConv2d(out_size//n_boxes,n_boxes,max_input_h,max_input_w,reparametrization_factor=reparam_factor), 
                                           nn.BatchNorm2d(out_size),#nn.Dropout(p = 0.5)
                                           
                )
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size//n_boxes, kernel_size = 1, stride = 1,padding = 0),
                                           BoxConv2d(out_size//n_boxes,n_boxes,max_input_h,max_input_w,reparametrization_factor=reparam_factor), 
                                           nn.ReLU()
                )
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size//n_boxes, kernel_size = 1, stride = 1,padding = 0),
                                           BoxConv2d(out_size//n_boxes,n_boxes,max_input_h,max_input_w,reparametrization_factor=reparam_factor), 
                                           nn.ReLU())
        else:
            if is_batchnorm:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
                )
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
                self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())
        
        if use_se == True and use_prelu == True:
            self.se_layer1 = ChannelSELayer(out_size, act = 'prelu')
            self.se_layer2 = ChannelSELayer(out_size, act = 'prelu')
        elif use_se == True and use_prelu == False:
            self.se_layer1 = ChannelSELayer(out_size, act = 'relu')
            self.se_layer2 = ChannelSELayer(out_size, act = 'relu')
        else:
            self.se_layer1 = None
            self.se_layer2 = None

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        if self.se_layer1 is not None:
            outputs = self.se_layer1(outputs)
        outputs = self.conv2(outputs)
        if self.se_layer2 is not None:
            outputs = self.se_layer2(outputs)
        return outputs



class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class unetUp(nn.Module):
    '''
    U-Net decoder block with default kernel size of 3
    
    Parameters:
        in_size     -- number of input channels
        out_size    -- number of output channels
        is_deconv   -- boolean flag to indicate if interpolation or de-convolution
                        should be used for up-sampling
    '''
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        
        self.conv = unetConv2(in_size, out_size, False, use_boxconv=False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))



class unet(nn.Module):
    '''
    U-Net architecture
    
    Parameters:
        in_channels     -- number of input channels
        out_channels    -- number of output channels
        feature_scale   -- scale for scaling default filter range in U-Net (default: 2)
        is_deconv       -- boolean flag to indicate if interpolation or de-convolution
                            should be used for up-sampling
        is_batchnorm    -- boolean flag to indicate batch-normalization usage
    ''' 
    def __init__(self, in_channels=3, out_channels = 21, feature_scale=1, is_deconv=True, is_batchnorm=True, max_input_h=64, max_input_w=64, use_boxconv=False):
        super(unet, self).__init__()
        
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        #self.h, self.w = max_input_h, max_input_w
        self.use_boxconv = use_boxconv
        
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        
        # downsampling
        #max_input_h = max_input_h //2
        #max_input_w = max_input_w //2
       
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm,  max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2
           
        #self.center = unetConv2(filters[3], filters[4], self.is_batchnorm,use_boxconv=self.use_boxconv, max_input_h=max_input_h, max_input_w=max_input_w)
        self.center = Bottleneck(filters[3], filters[4])


            
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

class unetm(nn.Module):
    '''
    mini U-Net architecture with 2 downsampling & upsampling blocks and one bottleneck
    with Squeeze and Excitation layers
    
    Parameters:
        in_channels     -- number of input channels
        out_channels    -- number of output channels
        feature_scale   -- scale for scaling default filter range in U-Net (default: 2)
        is_deconv       -- boolean flag to indicate if interpolation or de-convolution
                            should be used for up-sampling
        is_batchnorm    -- boolean flag to indicate batch-normalization usage
        use_SE          -- boolean flag to indicate SE blocks usage
        use_PReLU       -- boolean flag to indicate activation between linear layers in SE 
                            (relu vs. prelu)
    '''
    def __init__(self, in_channels=3, out_channels = 21, feature_scale=1, 
                 is_deconv=True, is_batchnorm=True, use_SE = False, use_PReLU = False, max_input_h=64, max_input_w=64, use_boxconv=False):
        super(unetm, self).__init__()
        
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.use_SE = use_SE
        self.use_PReLU = use_PReLU
        self.use_boxconv = use_boxconv
        filters = [64, 128, 256, 512, 1024]
#        filters = [128, 256, 512, 1024, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        #max_input_h=max_input_h  //2
        #max_input_w=max_input_w // 2
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h // 2
        max_input_w = max_input_w // 2

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU,  max_input_h=max_input_h, max_input_w=max_input_w)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        max_input_h = max_input_h //2 
        max_input_w = max_input_w //2 

        self.center = unetConv2(filters[1], filters[2], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU, use_boxconv=self.use_boxconv, max_input_h=max_input_h, max_input_w=max_input_w)

        # upsampling
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], out_channels, 1)

        
    
        
    def forward(self, inputs):
        
        conv1 = F.relu(self.conv1(inputs))
        maxpool1 = self.maxpool1(conv1)

        conv2 = F.relu(self.conv2(maxpool1))
        maxpool2 = self.maxpool2(conv2)

        center = self.center(maxpool2)
        up2 = self.up_concat2(conv2, center)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final