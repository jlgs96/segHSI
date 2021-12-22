import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
if __name__ == "__main__":
    

    listfs   = ["0.5", "1", "2"]

    models   = ['unet', 'unet_BOXCONV']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices','IOUs']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices']
    temetrics = ['test_losses', 'teoas', 'tempcas', 'temIOUs','tedices']
    valmetrics = ['val_losses', 'valoas', 'valmpcas', 'valmIOUs','valdices']
    
    n_epochs = 60
    #n_seeds = 5
    n_seeds = 5
    data = np.ones((len(models), len(listfs), n_seeds, len(valmetrics), n_epochs)) * -1000.0

    for fs in listfs:
        for model in models:
            
            for seed in range(5):
                npzFile= np.load("./NPZs/Validation/" + model + '_fs' + str(fs) + '_' + str(seed)+'_'+ 'VAL' + '.npz')
                idfs    = listfs.index(fs)
                idmodel = models.index(model)
                #for idmet, met in enumerate(metrics):
                data[idmodel, idfs, seed, :, :] = npzFile['valData']

    pos_max = np.argmax(data[:,:,:,3,:],axis = 3)
    


    for fs in listfs:
        for model in models:
            for seed in range(5):
                npzFile= np.load("./NPZs/Test/"+ model + '_fs' + str(fs) + '_' + str(seed)+'_'+ 'TE' + '.npz')
                idfs    = listfs.index(fs)
                idmodel = models.index(model)
                #for idmet, met in enumerate(metrics):
                data[idmodel, idfs, seed, :, :] = npzFile['teData']



    data_max = np.ones((len(models), len(listfs), n_seeds, len(valmetrics))) * -1000.0
    #for idmet, met in enumerate(['Te. Loss','teOA(\%)', 'teAA(\%)', 'temIOUs','temDICES']):
    for idfs, fs in enumerate(listfs):
        for (idmodel, model), namelegend in zip(enumerate(models), ["UNET", "UNETBX"]):
            for seed in range(5):
                data_max[idmodel, idfs, seed, :] = data[idmodel, idfs, seed, :, pos_max[idmodel, idfs, seed]]

    data_avg = np.average(data_max, axis=2)
    data_std = np.std(data_max, axis=2)


    for idfs, fs in enumerate(listfs):
        for (idmodel, model), namelegend in zip(enumerate(models), ["UNET", "UNETBX"]):
            string = model + " & "
            for idmet, met in enumerate(temetrics):
                if idmet in [0]: # estas son en tanto por 1
                    string += str(np.round(data_avg[idmodel, idfs, idmet], 2)) + " & "
                else:
                    string += str(np.round(data_avg[idmodel, idfs, idmet]*100.0, 2)) + " & "
            print(string[:-2] + r"\n")
            
    
