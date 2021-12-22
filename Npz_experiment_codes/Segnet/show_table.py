import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
if __name__ == "__main__":
    
    models   = ['segnet', 'segnet_BOXCONV']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices','IOUs']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices']
    temetrics = ['test_losses', 'teoas', 'tempcas', 'temIOUs','tedices']
    valmetrics = ['val_losses', 'valoas', 'valmpcas', 'valmIOUs','valdices']
    
    n_epochs = 60
    #n_seeds = 5
    n_seeds = 5
    data = np.ones((len(models),n_seeds, len(valmetrics), n_epochs)) * -1000.0
    for model in models:
        for seed in range(5):
            npzFile= np.load("./NPZs/Val/" + model + '_' + str(seed)+'_'+ 'VAL' + '.npz')
            idmodel = models.index(model)
            #for idmet, met in enumerate(metrics):
            data[idmodel,seed, :, :] = npzFile['valData']

    pos_max = np.argmax(data[:,:,3,:],axis = 2)
    


    for model in models:
        for seed in range(5):
            npzFile= np.load("./NPZs/Test/"+ model + '_' + str(seed)+'_'+ 'TE' + '.npz')
           
            idmodel = models.index(model)
            #for idmet, met in enumerate(metrics):
            data[idmodel,seed, :, :] = npzFile['teData']




    data_max = np.ones((len(models),n_seeds, len(valmetrics))) * -1000.0
    for (idmodel, model), namelegend in zip(enumerate(models), ["segnet", "segnetBX"]):
        for seed in range(5):
            data_max[idmodel,seed, :] = data[idmodel,seed, :, pos_max[idmodel, seed]]

    data_avg = np.average(data_max, axis=2)
    data_std = np.std(data_max, axis=2)


    for (idmodel, model), namelegend in zip(enumerate(models), ["segnet", "segnetBX"]):
        string = model + " & "
        for idmet, met in enumerate(temetrics):
            if idmet in [0]: # estas son en tanto por 1
                string += str(np.round(data_avg[idmodel, idmet], 2)) + " & "
            else:
                string += str(np.round(data_avg[idmodel,idmet]*100.0, 2)) + " & "
        print(string[:-2] + r"\n")
        
    
