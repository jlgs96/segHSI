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
#from networks2.resnet62 import ResnetGenerator
#from networks2.ERFNet import ERFNet, BoxERFNet
#from networks.modelsboxconv import ENet, BoxENet
from networks import modelsboxconv
from networks.segnet import segnet as aeroSegnet
import argparse
import matplotlib.pyplot as plt
# Define a manual seed to help reproduce identical results
#torch.manual_seed(3108)







def train(epoch = 0, show_iter=5):
    
    global trainloss
    trainloss2 = AverageMeter()
    
    
    
    
    
    print('\nTrain Epoch: %d' % epoch)
    
    
    net.train()

    running_loss = 0.0
    
    for idx, (rgb_ip, hsi_ip, labels) in enumerate(trainloader, 0):
        #print(rgb_ip.shape, hsi_ip.shape, labels.shape)
#        print(idx)
        N = hsi_ip.size(0)
        optimizer.zero_grad()
        
        outputs = net(hsi_ip.to(device))
        
        loss = criterion(outputs, labels.to(device))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        trainloss2.update(loss.item(), N)
        #train_losses.append(loss.item())

        if (idx + 1) %  show_iter == 0:
            print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / show_iter))
            running_loss = 0.0
    trainloss.append(trainloss2.avg)
    train_losses.append(trainloss2.avg)
    output_f.write(str('TR loss: %.3f' % (trainloss2.avg))+ '\n')
    
    print('TR loss: %.3f' % (trainloss2.avg))
       
        

def val(epoch = 0):
    
    global valloss
    valloss2 = AverageMeter()
    truth = []
    pred = []
    
    print('\nVal Epoch: %d' % epoch)
    
    
    net.eval()

    valloss_fx = 0.0
    
    with torch.no_grad():
        for idx, (rgb_ip, hsi_ip, labels) in enumerate(valloader, 0):
    #        print(idx)
            N = hsi_ip.size(0)
            
            outputs = net(hsi_ip.to(device))
            
            loss = criterion(outputs, labels.to(device))
            
            valloss_fx += loss.item()    
    

            
            valloss2.update(loss.item(), N)
            #val_losses.append(loss.item())
            truth = np.append(truth, labels.cpu().numpy())
            pred = np.append(pred, outputs.max(1)[1].cpu().numpy())
            
            print("{0:.2f}".format((idx+1)*100/(len(valset))), end = '-', flush = True)
    print()
    output_f2.write(str('VAL: %d loss: %.3f' % (epoch + 1, valloss_fx / (idx+1)))+ '\n')
    print('VAL: %d loss: %.3f' % (epoch + 1, valloss_fx / (idx+1)))
    valloss.append(valloss2.avg)
    val_losses.append(valloss2.avg)
    return perf(truth, pred)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    
    output_f = open("trainLoss.txt","a")
    output_f2 = open("VallLoss.txt","a")
  
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
    print(seeds[args.idtest])
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



    if (args.use_boxconv):
        output_f2.write("-------------UNET_Boxconv_Validation-------------" + '\n')
        output_f.write("-------------UNET_Boxconv_train-------------"+ '\n')
    else:
        output_f2.write("-------------UNET_Validation-------------"+ '\n')
        output_f.write("-------------UNET_train-------------"+ '\n')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
    valloader = torch.utils.data.DataLoader(valset, batch_size = args.batch_size, shuffle = False)
    
    #Pre-computed weights using median frequency balancing    
    weights = [1.11, 0.37, 0.56, 4.22, 6.77, 1.0]
    weights = torch.FloatTensor(weights)
    
    criterion = cross_entropy2d(reduction = 'mean', weight=weights.cuda(), ignore_index = 5)
    
    if args.network_arch.lower() == 'resnet':
        
        net = ResnetGeneratorBX(args.bands, 6, n_blocks=args.resnet_blocks, use_boxconv=args.use_boxconv, use_head_box=args.use_head_box)
        
    elif args.network_arch.lower() == 'segnet':
        if args.use_mini == True:
            net = segnetm(args.bands, 6)
        else:
            net = SegNet(args.bands,6, 4, use_boxconv=args.use_boxconv)
            #net = aeroSegnet(args.bands,6, 4, use_boxconv=args.use_boxconv)
            #net = aeroSegnet(args.bands,6)
            
    elif args.network_arch.lower() == 'unet':
        if args.use_mini == True:
            net = unetm(args.bands, 6, use_boxconv=args.use_boxconv, use_SE = args.use_SE, use_PReLU = args.use_preluSE, feature_scale=args.feature_scale)
        else:
            net = unet(args.bands, 6, use_boxconv=args.use_boxconv, use_SE = args.use_SE, use_PReLU = args.use_preluSE, feature_scale=args.feature_scale)
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
    print(net)
    print(count_parameters(net))
    #print(args.network_arch + "," + str(count_parameters(net)))
    #output_f.write(args.network_arch + "," + str(count_parameters(net))+ ", ")

    #exit()
    
    
    init_weights(net, init_type=args.init_weights)
    
    
    if args.pretrained_weights is not None:
        load_weights(net, args.pretrained_weights)
        print('Completed loading pretrained network weights')
        
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,50])

    trainloss = []
    valloss = []
    bestmiou = 0
    train_losses = []
    val_losses = []
       
    for epoch in range(args.epochs):
        train(epoch)
        
        oa, mpca, mIOU, dice, IOU = val(epoch)
        
      
        
        print('Overall acc  = {:.3f}, MPCA = {:.3f}, mIOU = {:.3f}'.format(oa, mpca, mIOU))
       
        #output_f.write('Overall acc  = {:.3f}, MPCA = {:.3f}, mIOU = {:.3f}\n'.format(oa, mpca, mIOU))
        if mIOU > bestmiou:
            bestmiou = mIOU
            torch.save(net.state_dict(), args.network_weights_path)
        scheduler.step()
    output_f2.close()
    output_f.close()
    #plt.figure(figsize=(10,5))
    #plt.title("Training and Validation Loss")
    #plt.plot(val_losses,label="val", color = "green",lw=1,alpha=0.8)
    ##plt.plot(x = 'epochs', y = 'Val losses', color = 'green', alpha=0.8, legend='Val loss', line_width=2,source=source)
    #plt.plot(train_losses,label="train", color = "blue",lw=1,alpha=0.8)
    ##plt.plot(x = 'epochs', y = 'Train losses', color = 'blue', alpha=0.8, legend='Train loss', line_width=2,source=source)
    #plt.xlabel("epochs")
    #plt.ylabel("Loss")
    #plt.legend()
    #plt.show()
