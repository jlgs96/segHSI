import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
#numFileTest=np.load('pruebanumpy.npz')
    #print(numFileTest['train'])
    #print(numFileTest['validation'])









if __name__ == "__main__":
    
    #parser = argparse.ArgumentParser(description = 'Plotlines for train/validation loss evaluation')
    ###NAME OF THE FILE TO LOAD
    #parser.add_argument('--file_name', default = None, help = 'Path to the NPZ file to load')
    #args = parse_args(parser)
    
    models   = ['unet', 'unet_BOXCONV']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices','IOUs']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices']
    metrics = ['test_losses', 'teoas', 'tempcas', 'temIOUs','tedices']

    n_epochs = 60
    n_seeds = 5
    data = np.ones((len(models), n_seeds, len(metrics), n_epochs)) * -1000.0

    for model in models:
        for seed in range(n_seeds):
            fs = 0.5 if model == "unet_BOXCONV" else 0.7
            npzFile= np.load(model + '_fs' + str(fs) + '_' + str(seed)+'_'+ 'TE' +  '.npz')
            idmodel = models.index(model)
            for idmet, met in enumerate(metrics):
                data[idmodel, seed, :, :] = npzFile['teData']


    data_avg = np.max(data, axis=1)
    #for i in range(4):
        #print(100.0 * data[1, i, 4, :])
    #exit()

    data_avg = np.average(data, axis=1)
    data_std = np.std(data, axis=1)


    #print(["model", 'Tr. Loss','Val. Loss', 'OA(\%)', 'AA(\%)', 'mIOUs','mDICES'])
    #for (idmodel, model), namelegend in zip(enumerate(models), ["UNET", "UNETBX"]):
        #string = model + " & "
        #for idmet, met in enumerate(['Te. Loss','teOA(\%)', 'teAA(\%)', 'temIOUs','temDICES']):
            #if idmet < 2:
                #string += str(np.round(np.min(data_avg[idmodel, :, :]), decimals=2))  + " & "
            #else:
                #string += str(np.round(100.0 * np.max(data_avg[idmodel, :, :]), decimals=2))  + " & "
        #print(string[:-2])


    for idmet, met in enumerate(['Te. Loss','teOA(\%)', 'teAA(\%)', 'temIOUs','temDICES']):
        for (idmodel, model), namelegend in zip(enumerate(models), ["UNET", "UNETBX"]):
            avg = data_avg[idmodel, idmet]
            std = data_std[idmodel, idmet]
            plt.plot(avg, label= met + " " + namelegend)
            plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
            
        if idmet == 0: continue
        #plt.figure(figsize=(10,5))
        plt.title("Training and Validation "+ met)
        #plt.plot(val_losses,label="Validation", color = "green",lw=1,alpha=0.8)
        #plt.plot(x = 'epochs', y = 'Val losses', color = 'green', alpha=0.8, legend='Val loss', line_width=2,source=source)
        #plt.plot(train_losses,label="Training", color = "blue",lw=1,alpha=0.8)
        #plt.plot(x = 'epochs', y = 'Train losses', color = 'blue', alpha=0.8, legend='Train loss', line_width=2,source=source)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


    
