#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:56:02 2019

@author: aneesh
"""

import os
import os.path as osp

import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np


from helpers.utils import Metrics, AeroCLoader, parse_args
from networks.resnet6 import ResnetGenerator
#from networks.unet import unet, unetm
#from networks.segnet_seg import segnet
#from networks.modelsboxconv import SegNet
from networks2.unet2 import unet, unetm
from networks2.segnet import SegNet
from networks import modelsboxconv
from networks2.ERFNet import ERFNet, BoxERFNet
import argparse

if __name__ == "__main__":
    #output_f = open("output_test.txt","a")
    parser = argparse.ArgumentParser(description = 'AeroRIT baseline evalutions')    
    
    ### 0. Config file?
    parser.add_argument('--config-file', default = None, help = 'Path to configuration file')
    
    ### 1. Data Loading
    parser.add_argument('--bands', default = 51, help = 'Which bands category to load \
                        - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type = int)
    parser.add_argument('--hsi_c', default = 'rad', help = 'Load HSI Radiance or Reflectance data?')
    
    ### 2. Network selections
    ### a. Which network?
    parser.add_argument('--network_arch', default = 'BoxEnet',\
        choices=["segnet","unet","resnet","enet", "boxenet", "boxonlyenet","erfnet","boxerfnet"],type = lambda s : s.lower(), help = 'Network architecture?')
    parser.add_argument('--use_mini', action = 'store_true', help = 'Use mini version of network?')
    
    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default = 6, help = 'How many blocks if ResNet architecture?', type = int)
    
    ### c. UNet configs
    parser.add_argument('--use_SE', action = 'store_true', help = 'Network uses SE Layer?')
    parser.add_argument('--use_preluSE', action = 'store_true', help = 'SE layer uses ReLU or PReLU activation?')
    
    ### Load weights post network config
    parser.add_argument('--network_weights_path', default = "./savedmodels/Seg_Net_Boxconv.pt", help = 'Path to Saved Network weights')
    
    ### Use GPU or not
    parser.add_argument('--use_cpu', action = 'store_true', help = 'use GPUs?')


    parser.add_argument('--use_boxconv', action='store_true', help='Use box convolutions modules')
    parser.add_argument('--feature_scale', default = 4, help = 'feature_scale in different grades', type = float)
    parser.add_argument('--idtest', default = 0, help = 'id test', type = int)
    parser.add_argument('--use_head_box', action='store_true', help='Use head_box convolutions modules')
    args = parse_args(parser)
    print(args)
    device='cuda'
    if args.use_cpu:
        device = 'cpu'
    
    perf = Metrics()
    
    tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    
    if args.bands == 3 or args.bands == 4 or args.bands == 6:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = '{}b'.format(args.bands), transforms = tx)
    elif args.bands == 10:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'all', transforms = tx)
    elif args.bands == 31:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'visible', transforms = tx)
    elif args.bands == 51:
        #testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'all', transforms = tx)
        testset = AeroCLoader(set_loc = 'right', set_type = 'train', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'all', transforms = tx)
    else:
        raise NotImplementedError('required parameter not found in dictionary')
    
    print('Completed loading data...')
    
    output_f = open("output_test_" + args.network_arch + ".txt","a")
    if args.network_arch == 'resnet':
        output_f.write("Resnet"+",")
        net = ResnetGenerator(args.bands, 6, n_blocks=args.resnet_blocks, use_boxconv=args.use_boxconv,use_head_box= args.use_head_box)
    elif args.network_arch == 'segnet':
        output_f.write("Segnet"+",")
        if args.use_mini == True:
            net = segnetm(args.bands, 6)
        else:
            #net = segnet(args.bands, 6)
             net = SegNet(args.bands,6, 4, use_boxconv=args.use_boxconv)
    elif args.network_arch == 'unet':
        output_f.write("Unet"+",")
        if args.use_mini == True:
            net = unetm(args.bands, 6,use_boxconv=args.use_boxconv, use_SE = args.use_SE, use_PReLU = args.use_preluSE, feature_scale=args.feature_scale)
        else:
            net = unet(args.bands, 6, use_boxconv=args.use_boxconv, use_SE = args.use_SE, use_PReLU = args.use_preluSE, feature_scale=args.feature_scale)
    elif args.network_arch.lower() == 'enet':
        output_f.write("Enet"+",")
        args.pretrained_weights = None
        if args.use_mini == True:
            net = modelsboxconv.ENetPequena(n_bands = args.bands, n_classes = 6)
        else:
            net = modelsboxconv.ENet(n_bands = args.bands, n_classes = 6)
    elif args.network_arch.lower() == 'boxenet':
        output_f.write("BoxEnet"+",")
        args.pretrained_weights = None
        
        if args.use_mini == True:
            net = modelsboxconv.BoxENetPequena(n_bands = args.bands, n_classes = 6)
        else:
            
            net = modelsboxconv.BoxENet(n_bands = args.bands, n_classes = 6)      
    elif args.network_arch.lower() == 'boxonlyenet':
        output_f.write("BoxOnlyENet"+",")
        args.pretrained_weights = None
        if args.use_mini == True:
            net = modelsboxconv.BoxOnlyENetPequena(n_bands = args.bands, n_classes = 6)
        else:
            net = modelsboxconv.BoxOnlyENet(n_bands = args.bands, n_classes = 6)
    elif args.network_arch.lower() == 'erfnet':
        output_f.write("ERFNet"+",")
        net = ERFNet(n_bands = args.bands, n_classes = 6)
    elif args.network_arch.lower() == 'boxerfnet':
        output_f.write("BoxERFNet"+",")
        net = BoxERFNet(n_classes = 6)        
    else:
        raise NotImplementedError('required parameter not found in dictionary')

    print(net)
    
    from box_convolution import BoxConv2d
    net.load_state_dict(torch.load(args.network_weights_path))
    net.eval()
    net.to(device)
    
    print('Completed loading pretrained network weights...')
    
    print('Calculating prediction accuracy...')
    
    labels_gt = []
    labels_pred = []
    
    for img_idx in range(len(testset)):
        _, hsi, label = testset[img_idx]
        label = label.numpy()
        
        label_pred = net(hsi.unsqueeze(0).to(device))
        label_pred = label_pred.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        
        label = label.flatten()
        label_pred = label_pred.flatten()
        
        labels_gt = np.append(labels_gt, label)
        labels_pred = np.append(labels_pred, label_pred)
    
    scores = perf(labels_gt, labels_pred)
    
    
    
    print('Statistics on Test set:\n')
    #output_f.write('Statistics on Test set:\n')
    print('Overall accuracy = {:.2f}%\nAverage Accuracy = {:.2f}%\nMean IOU is {:.2f}\
          \nMean DICE score is {:.2f}'.format(scores[0]*100, scores[1]*100, scores[2]*100, scores[3]*100))
    #output_f.write('Overall accuracy = {:.2f}%\nAverage Accuracy = {:.2f}%\nMean IOU is {:.2f}\
          #\nMean DICE score is {:.2f}'.format(scores[0]*100, scores[1]*100, scores[2]*100, scores[3]*100))

    #output_f.write(str(args.use_mini) + "," + str(args.use_SE) + "," + str(args.use_preluSE) + "," + str(args.use_boxconv) + "," + \
                   #str(args.idtest) + "," + str(args.feature_scale)+ "," + str(scores[0]*100.0)+ "," +str(scores[1]*100.0)+ "," +str(scores[2]*100.0)+ "," +str(scores[3]*100.0) + '\n')
    #print(args.network_arch + "," + str(args.use_mini) + "," + str(args.use_SE) + "," + str(args.use_preluSE) + "," + str(args.use_boxconv) + "," + \
                    #str(args.idtest) + "," + str(args.feature_scale) + "," + str(scores[0]*100.0)+ "," +str(scores[1]*100.0)+ "," +str(scores[2]*100.0)+ "," +str(scores[3]*100.0) + '\n')
    #output_f.close()


    args.feature_scale = args.resnet_blocks
    output_f.write(str(args.use_mini) + "," + str(args.use_SE) + "," + str(args.use_head_box) + "," + str(args.use_boxconv) + "," + \
                   str(args.idtest) + "," + str(args.feature_scale)+ "," + str(scores[0]*100.0)+ "," +str(scores[1]*100.0)+ "," +str(scores[2]*100.0)+ "," +str(scores[3]*100.0) + '\n')
    print(args.network_arch + "," + str(args.use_mini) + "," + str(args.use_SE) + "," + str(args.use_head_box) + "," + str(args.use_boxconv) + "," + \
                    str(args.idtest) + "," + str(args.feature_scale) + "," + str(scores[0]*100.0)+ "," +str(scores[1]*100.0)+ "," +str(scores[2]*100.0)+ "," +str(scores[3]*100.0) + '\n')
    output_f.close()

