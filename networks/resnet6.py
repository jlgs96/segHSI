#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 06:00:09 2019

@author: aneesh
"""

import torch
import torch.nn as nn

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
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

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

        mult = 2**n_downsampling
        r = 0.510; h = 64; w = 64
        for i in range(n_blocks):
            #ESTO FUE LOS DIFERENTES BLOQUES QUE SE AÑADIERON, PROBANDO EL QUE MEJOR FUE ERA EL ÚLTIMO.
            
            ###RESNET BLOCK NORMAL CON BOTTLENECK NORMAL###
            #model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]
            #model += [Bottleneck(ngf * mult, ngf * mult, use_dropout=use_dropout)]
            ###RESNETBLOCKBOXVON PRUEBA 1 Y PRUEBA 2 CON BOTTLENECK###
            #model += [ResnetBlockBoxConv(ngf * mult, 4, h // n_downsampling, w // n_downsampling, 0.4, reparam_factor=r)]
            #model += [ResnetBlockBoxConv(ngf * mult, 4, h, w, 0.4, reparam_factor=r),
                        #Bottleneck(ngf * mult, ngf * mult, use_dropout=True, asymmetric_ksize=5)]
            ###RESNET BLOCK BOXCONV PRUEBA FINAL CON BOTTLENECK###
            model += [ResnetBlockBoxConv(ngf * mult, 4, h // n_downsampling, w // n_downsampling, 0.4, reparam_factor=r),
                        Bottleneck(ngf * mult, ngf * mult, use_dropout=True, asymmetric_ksize=5)]


        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]


        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


        mult = 2**n_downsampling
        r = 0.510; h = 64; w = 64
        for i in range(n_blocks):
             ###RESNET BLOCK NORMAL CON BOTTLENECK NORMAL###
            #model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]
            #model += [Bottleneck(ngf * mult, ngf * mult, use_dropout=use_dropout)]
            
            ###RESNETBLOCKBOXVON PRUEBA 1###
            #model += [ResnetBlockBoxConv(ngf * mult, 4, h // n_downsampling, w // n_downsampling, 0.4, reparam_factor=r)]
            
            ###RESNET BLOCK BOXCONV PRUEBA FINAL CON BOTTLENECK###
            model += [ResnetBlockBoxConv(ngf * mult, 4, h , w , 0.4, reparam_factor=r)]
            model += [Bottleneck(128, 128, use_dropout=True, asymmetric_ksize=5)]



from box_convolution import BoxConv2d

class ResnetBlockBoxConv(nn.Module):
    def __init__(self, in_channels, num_boxes, max_input_h, max_input_w,
        dropout_prob=0.0, reparam_factor=1.5625):

        super().__init__()
        assert in_channels % num_boxes == 0
        bt_channels = in_channels // num_boxes # bottleneck channels

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1,1), bias=False),
            nn.BatchNorm2d(bt_channels),
            nn.ReLU(True),
            
            # BEHOLD:
            BoxConv2d(
                bt_channels, num_boxes, max_input_h, max_input_w,
                reparametrization_factor=reparam_factor),

            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout_prob))

    def forward(self, x):
        return (x + self.main_branch(x)).relu_()


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, downsample=False,
        asymmetric_ksize=None, dilation=1, use_prelu=True):
        if use_dropout: dropout_prob = 0.1
        else: dropout_prob = 0.0

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
            nn.BatchNorm2d(out_channels, 1e-3),
            nn.Dropout2d(dropout_prob)
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
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias = False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
