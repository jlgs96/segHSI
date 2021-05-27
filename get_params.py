#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:34:16 2019

@author: aneesh
"""

import os 
import os.path as osp

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import optim

from helpers.augmentations import RandomHorizontallyFlip, RandomVerticallyFlip, \
    RandomTranspose, Compose
from helpers.utils import AeroCLoader, AverageMeter, Metrics, parse_args
from helpers.lossfunctions import cross_entropy2d
from torchvision import transforms
from networks.resnet6 import ResnetGenerator as ResnetGeneratorBX
from networks.model_utils import init_weights, load_weights 
from networks2.unet2 import unet, unetm
from networks2.segnet import SegNet
from networks2.resnet62 import ResnetGenerator
from networks2.ERFNet import ERFNet, BoxERFNet
#from networks.modelsboxconv import ENet, BoxENet
from networks import modelsboxconv
import argparse

# Define a manual seed to help reproduce identical results
#torch.manual_seed(3108)






def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    
    #output_f = open("output_test.txt","a")
    
    parser = argparse.ArgumentParser(description = 'AeroRIT baseline evalutions')
    
    ### 0. Config file?
    #parser.add_argument('--config-file', default = None, help = 'Path to configuration file')
    ##PRUEBA PARA ERROR POSIBLE QUE NO HAYA CARGADO EL PATH AL CONFIG FILE ASI QUE LO PONGO POR DEFECTO TENIENDO EN CUENTA QUE ES UNET
 


    
    parser.add_argument('--config-file', default = None, help = 'Path to configuration file')
    
    
    
    ### 1. Data Loading
    parser.add_argument('--bands', default = 51, help = 'Which bands category to load \
                        - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type = int)
    parser.add_argument('--hsi_c', default = 'rad', help = 'Load HSI Radiance or Reflectance data?')
    parser.add_argument('--use_augs', action = 'store_true', help = 'Use data augmentations?')
    
   ### 2. Network selections
    ### a. Which network?
    
    ###POR DEFECTO SELECCIONAREMOS RESNET QUE ES LA QUE ESTÁ MODIFICADA###
    parser.add_argument('--network_arch', default ='ResNet',\
        choices=["segnet","unet","resnet","enet", "boxenet", "boxonlyenet","erfnet","boxerfnet"], type = lambda s : s.lower(), help = 'Network architecture?')
    parser.add_argument('--use_mini', action = 'store_true', help = 'Use mini version of network?')
    
    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default = 6, help = 'How many blocks if ResNet architecture?', type = int)
    
    ### c. UNet configs
    parser.add_argument('--use_SE', action = 'store_true', help = 'Network uses SE Layer?')
    parser.add_argument('--use_preluSE', action = 'store_true', help = 'SE layer uses ReLU or PReLU activation?')
    
    ### Save weights post network config
    parser.add_argument('--network_weights_path', default = "./savedmodels/Seg_Net_Boxconv.pt", help = 'Path to save Network weights')
    
    ### Use GPU or not
    parser.add_argument('--use_cpu', action = 'store_true', help = 'use GPUs?')

    ### Use GPU or not
    parser.add_argument('--use_myargs', action = 'store_true', help = 'use default config?')

    
    ### Hyperparameters
    parser.add_argument('--batch-size', default = 100, type = int, help = 'Number of images sampled per minibatch?')
    parser.add_argument('--init_weights', default = 'kaiming', help = "Choose from: 'normal', 'xavier', 'kaiming'")
    parser.add_argument('--learning-rate', default = 1e-4, type = float, help = 'Initial learning rate for training the network?')
    parser.add_argument('--epochs', default = 60, type = int, help = 'Maximum number of epochs?')
    
    ### Pretrained representation present?
    parser.add_argument('--pretrained_weights', default = None, help = 'Path to pretrained weights for network')

    parser.add_argument('--use_boxconv', action='store_true', help='Use box convolutions modules')
    parser.add_argument('--idtest', default = 0, help = 'id test', type = int)

    parser.add_argument('--feature_scale', default = 4, help = 'feature_scale in different grades', type = float)
    parser.add_argument('--use_head_box', action='store_true', help='Use head_box convolutions modules')

    #parser.add_argument('--seed', default = 3108, help = 'random seed', type = int)
    
    args = parse_args(parser)

    seeds = [8451, 2262, 4618, 1232, 5920, 9473, 3108, 6799, 7774, 5315]
    #print(seeds[args.idtest])
    torch.manual_seed(seeds[args.idtest])
    
    #if args.use_myargs:
        #args = myconfig(args)
    #print(args)
    
    #if args.use_cuda and torch.cuda.is_available():
        #device = 'cuda'
    #else:
        #device = 'cpu'
    device = 'cuda'
    if args.use_cpu: device = 'cpu'
    
        
    
    perf = Metrics()
    
    if args.use_augs:
        augs = []
        augs.append(RandomHorizontallyFlip(p = 0.5))
        augs.append(RandomVerticallyFlip(p = 0.5))
        augs.append(RandomTranspose(p = 1))
        augs_tx = Compose(augs)
    else:
        augs_tx = None
        
    tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    
    if args.bands == 3 or args.bands == 4 or args.bands == 6:
        hsi_mode = '{}b'.format(args.bands)
    elif args.bands == 31:
        hsi_mode = 'visible'
    elif args.bands == 51 or args.bands == 10: #ALERT
        hsi_mode = 'all'
    else:
        raise NotImplementedError('required parameter not found in dictionary')
        
    trainset = AeroCLoader(set_loc = 'left', set_type = 'train', size = 'small', \
                           hsi_sign=args.hsi_c, hsi_mode = hsi_mode,transforms = tx, augs = augs_tx)
    #valset = AeroCLoader(set_loc = 'mid', set_type = 'test', size = 'small', \
                         #hsi_sign=args.hsi_c, hsi_mode = hsi_mode, transforms = tx)

    valset = AeroCLoader(set_loc = 'mid', set_type = 'train', size = 'small', \
                         hsi_sign=args.hsi_c, hsi_mode = hsi_mode, transforms = tx)




    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = args.batch_size, shuffle = False)
    
    #Pre-computed weights using median frequency balancing    
    weights = [1.11, 0.37, 0.56, 4.22, 6.77, 1.0]
    weights = torch.FloatTensor(weights)
    
    criterion = cross_entropy2d(reduction = 'mean', weight=weights.cuda(), ignore_index = 5)
    
    if args.network_arch.lower() == 'resnet':
        net = ResnetGeneratorBX(args.bands, 6, n_blocks=args.resnet_blocks, use_boxconv=args.use_boxconv, use_head_box=args.use_head_box)
        print(args.network_arch, args.resnet_blocks, args.use_boxconv, count_parameters(net))
    elif args.network_arch.lower() == 'segnet':
        if args.use_mini == True:
            net = segnetm(args.bands, 6)
        else:
            net = SegNet(args.bands,6, 4, use_boxconv=args.use_boxconv)
    elif args.network_arch.lower() == 'unet':
        if args.use_mini == True:
            net = unetm(args.bands, 6, use_boxconv=args.use_boxconv, use_SE = args.use_SE, use_PReLU = args.use_preluSE, feature_scale=args.feature_scale)
        else:
            net = unet(args.bands, 6, use_boxconv=args.use_boxconv, use_SE = args.use_SE, use_PReLU = args.use_preluSE, feature_scale=args.feature_scale)
        print(args.network_arch, args.feature_scale, args.use_boxconv, count_parameters(net))
    elif args.network_arch.lower() == 'enet':
        args.pretrained_weights = None
        if args.use_mini == True:
            net = modelsboxconv.ENet(n_bands = args.bands, n_classes = 6)
        else:
            net = modelsboxconv.ENet(n_bands = args.bands, n_classes = 6)
    elif args.network_arch.lower() == 'boxenet':
        args.pretrained_weights = None
        if args.use_mini == True:
            net = modelsboxconv.BoxENet(n_bands = args.bands, n_classes = 6)
        else:
            net = modelsboxconv.BoxENet(n_bands = args.bands, n_classes = 6)      
    elif args.network_arch.lower() == 'boxonlyenet':
        args.pretrained_weights = None
        if args.use_mini == True:
            net = modelsboxconv.BoxOnlyENetPequena(n_bands = args.bands, n_classes = 6)
        else:
            net = modelsboxconv.BoxOnlyENet(n_bands = args.bands, n_classes = 6)
    elif args.network_arch.lower() == 'erfnet':
            args.pretrained_weights == None
            net = ERFNet(n_bands = args.bands, n_classes = 6)
    elif args.network_arch.lower() == 'boxerfnet':
            args.pretrained_weights == None
            net = BoxERFNet(n_bands=args.bands, n_classes = 6)
    else:
        raise NotImplementedError('required parameter not found in dictionary')

