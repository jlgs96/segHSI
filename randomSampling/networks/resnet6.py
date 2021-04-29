#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 06:00:09 2019

@author: aneesh
"""

import torch
import torch.nn as nn
from box_convolution import BoxConv2d

class ResnetGenerator(nn.Module):
    '''
    Construct a Resnet-based generator
    Ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    
    Parameters:
        input_nc (int)      -- the number of channels in input images
        output_nc (int)     -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers
        n_blocks (int)      -- the number of ResNet blocks
        gpu_ids             -- GPUs for parallel processing
    '''
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, max_input_h=64, max_input_w=64, use_boxconv = False, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.h, self.w = max_input_h, max_input_w
        self.use_boxconv = use_boxconv
        
        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                    norm_layer(ngf * mult * 2, affine=True),
                    nn.ReLU(True)]
            max_input_h, max_input_w = max_input_h// 2, max_input_w// 2
            
            

        mult = 2**n_downsampling
        for i in range(n_blocks):
            
            if(1==1):
                model += \
                    [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout, max_input_h=self.h, max_input_w=self.w, use_boxconv=self.use_boxconv)]
            else:
                 model += \
                    [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout, max_input_h=self.h, max_input_w=self.w)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
        
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1),
                    norm_layer(int(ngf * mult / 2), affine=True),
                    nn.ReLU(True)]
            max_input_h, max_input_w = max_input_h * 2, max_input_w * 2
        

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class ResnetBlock(nn.Module):
    '''
    Defines a ResNet block
    Ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    
    Parameters:
        dim (int)           -- the number of channels in the conv layer.
        padding_type (str)  -- the name of padding layer: reflect | replicate | zero
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers.
        use_bias (bool)     -- if the conv layer uses bias or not
    '''
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, max_input_h, max_input_w, use_bias = False, use_boxconv = False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, max_input_h, max_input_w, use_boxconv=use_boxconv)
        #self.conv_block = self.build_bottleneck(dim, padding_type, norm_layer, use_dropout, use_bias, max_input_h, max_input_w, use_boxconv=use_boxconv)


    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, max_input_h, max_input_w, use_boxconv=False):
        conv_block = []
        p = 0
        reparam_factor = 1.5625
        n_boxes = 4
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        
        if use_boxconv:
            conv_block += [nn.Conv2d(dim, dim//n_boxes, kernel_size=1, stride = 1, padding = 0, bias=use_bias),norm_layer(dim//n_boxes),
                    BoxConv2d(
                    dim//n_boxes, n_boxes, max_input_h, max_input_w,
                    reparametrization_factor=reparam_factor),
                    norm_layer(dim), nn.ReLU()]
           
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                        norm_layer(dim),
                        nn.ReLU(True)]            
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        if use_boxconv:
                    conv_block += [nn.Conv2d(dim, dim//n_boxes, kernel_size=1, stride = 1, padding = 0, bias=use_bias),norm_layer(dim//n_boxes), #
                            BoxConv2d(
                            dim//n_boxes, n_boxes, max_input_h, max_input_w,
                            reparametrization_factor=reparam_factor),
                            norm_layer(dim),
                            nn.ReLU()]
        else:  
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                            norm_layer(dim)]
        return nn.Sequential(*conv_block)


    def build_bottleneck(self, dim, padding_type, norm_layer, use_dropout, use_bias, max_input_h, max_input_w, use_boxconv=False):
            conv_block = []
            p = 0
            reduction_factor = 4
            reparam_factor = 0.860
            n_boxes = 4
    

            
            if use_boxconv:
                conv_block += [nn.Conv2d(dim, dim//n_boxes, kernel_size=1, stride = 1, padding = 0, bias=use_bias),norm_layer(dim//n_boxes),
                        BoxConv2d(
                        dim//n_boxes, n_boxes, max_input_h, max_input_w,
                        reparametrization_factor=reparam_factor),
                        norm_layer(dim), nn.ReLU()]
        
            else:
                conv_block += [nn.Conv2d(dim, dim//reduction_factor, kernel_size=1, padding=0, bias=use_bias),
                            norm_layer(dim//reduction_factor),
                            nn.Conv2d(dim//reduction_factor, dim//reduction_factor, kernel_size=3, padding=1, bias=use_bias),
                            nn.Conv2d(dim//reduction_factor, dim, kernel_size=1, padding=0, bias=use_bias),
                            norm_layer(dim),nn.ReLU()]            
            
            
            return nn.Sequential(*conv_block)

    def forward(self, x):      
        out = x + self.conv_block(x)
        return out

