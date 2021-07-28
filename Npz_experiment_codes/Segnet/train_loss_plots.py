import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
#numFileTest=np.load('pruebanumpy.npz')
    #print(numFileTest['train'])
    #print(numFileTest['validation'])






##PINTAR DIAGRAMA DE BARRAS PLT.BAR Y EN EJE X: FEATURESCALES Y EN EJE Y: miouMAX
##IGUAL PERO EN EJEX : NUM PARAMETROS Y EN Y MIOUMAX


if __name__ == "__main__":
    
    #parser = argparse.ArgumentParser(description = 'Plotlines for train/validation loss evaluation')
    ###NAME OF THE FILE TO LOAD
    #parser.add_argument('--file_name', default = None, help = 'Path to the NPZ file to load')
    #args = parse_args(parser)
    
    
    models   = ['segnet', 'segnet_BOXCONV']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices','IOUs']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices']
    metrics = ['test_losses', 'teoas', 'tempcas', 'temIOUs','tedices']
    n_epochs = 60
    #n_seeds = 5
    n_seeds = 5
    data = np.ones((len(models),n_seeds, len(metrics), n_epochs)) * -1000.0

    for model in models:        
        for seed in range(5):
            #INSERTAR AQUI CARPETA DONDE ESTEN LOS NPZS
            npzFile= np.load("/home/josel/Escritorio/Npz_experiment_codes/Segnet/NPZs/Test/" + model+'_' + str(seed)+'_'+ 'TE' + '.npz')
            idmodel = models.index(model)
            #for idmet, met in enumerate(metrics):
            data[idmodel,seed, :, :] = npzFile['teData']

                

    data_avg = np.average(data, axis=2)
    data_std = np.std(data, axis=2)


    #for idmet, met in enumerate(['Tr. Loss','Val. Loss', 'OA(\%)', 'AA(\%)', 'mIOUs','mDICES']):
    for idmet, met in enumerate(['Te. Loss','teOA(\%)', 'teAA(\%)', 'temIOUs','temDICES']):
        for (idmodel, model), namelegend in zip(enumerate(models), ["SEGNET", "SEGNETBX"]):
            avg = data_avg[idmodel,idmet]
            std = data_std[idmodel,idmet]
            plt.plot(avg, label= met + " " + namelegend)
            plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
            
        if idmet == 0:
            plt.ylim(0.2,1.2)
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
