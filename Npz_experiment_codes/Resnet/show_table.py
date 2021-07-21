import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
if __name__ == "__main__":
    

    listrb   = ["6", "9",]

    models   = ['resnet', 'resnet_BOXCONV']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices','IOUs']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices']
    temetrics = ['test_losses', 'teoas', 'tempcas', 'temIOUs','tedices']
    valmetrics = ['val_losses', 'valoas', 'valmpcas', 'valmIOUs','valdices']
    
    n_epochs = 60
    #n_seeds = 5
    n_seeds = 5
    data = np.ones((len(models), len(listrb), n_seeds, len(valmetrics), n_epochs)) * -1000.0

    for rb in listrb:
        for model in models:
            
            for seed in range(5):
                npzFile= np.load("/home/joseluis/segHSI/GRAFICAS/Resnet/NPZS/Val/" + model + '_RB' + str(rb) + '_' + str(seed)+'_'+ 'VAL' + '.npz')
                idrb    = listrb.index(rb)
                idmodel = models.index(model)
                #for idmet, met in enumerate(metrics):
                data[idmodel, idrb, seed, :, :] = npzFile['valData']

    pos_max = np.argmax(data[:,:,:,3,:],axis = 3)
    --use_boxconv


    for rb in listrb:
        for model in models:
            for seed in range(5):
                npzFile= np.load("/home/joseluis/segHSI/GRAFICAS/Resnet/NPZS/Test/"+ model + '_RB' + str(rb) + '_' + str(seed)+'_'+ 'TE' + '.npz')
                idrb    = listrb.index(rb)
                idmodel = models.index(model)
                #for idmet, met in enumerate(metrics):
                data[idmodel, idrb, seed, :, :] = npzFile['teData']



    data_max = np.ones((len(models), len(listrb), n_seeds, len(valmetrics))) * -1000.0
    #for idmet, met in enumerate(['Te. Loss','teOA(\%)', 'teAA(\%)', 'temIOUs','temDICES']):
    for idrb, rb in enumerate(listrb):
        for (idmodel, model), namelegend in zip(enumerate(models), ["RESNET", "RESNETBX"]):
            for seed in range(5):
                data_max[idmodel, idrb, seed, :] = data[idmodel, idrb, seed, :, pos_max[idmodel, idrb, seed]]

    data_avg = np.average(data_max, axis=2)
    data_std = np.std(data_max, axis=2)


    for idrb, rb in enumerate(listrb):
        for (idmodel, model), namelegend in zip(enumerate(models), ["RESNET", "RESNETBX"]):
            string = model + " & "
            for idmet, met in enumerate(temetrics):
                if idmet in [0]: # estas son en tanto por 1
                    string += str(np.round(data_avg[idmodel, idrb, idmet], 2)) + " & "
                else:
                    string += str(np.round(data_avg[idmodel, idrb, idmet]*100.0, 2)) + " & "
            print(string[:-2] + r"\n")
            
    
