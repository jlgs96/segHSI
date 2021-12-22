import numpy as np
import matplotlib.pyplot as plt
import argparse


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
    data = np.ones((len(models), n_seeds, len(metrics), n_epochs)) * -1000.0

    for model in models:        
        for seed in range(5):
            #INSERTAR AQUI CARPETA DONDE ESTEN LOS NPZS
            npzFile= np.load("./NPZs/Test/" + model+'_' + str(seed)+'_'+ 'TE' + '.npz')
            idmodel = models.index(model)
            #for idmet, met in enumerate(metrics):
            data[idmodel,seed, :, :] = npzFile['teData']
        
        #Uncomment this to use NPZs training files
        #for seed in range(5):
            ##INSERTAR AQUI CARPETA DONDE ESTEN LOS NPZS
            #npzFile= np.load("./NPZs/Train/" + model+'_' + str(seed)+'_'+ 'TR' + '.npz')
            #idmodel = models.index(model)
            ##for idmet, met in enumerate(metrics):
            #data[idmodel,seed, :, :] = npzFile['trData']

                

    data_avg = np.average(data, axis=1)
    data_std = np.std(data, axis=1)


    #for idmet, met in enumerate(['Tr. Loss','Val. Loss', 'OA(\%)', 'AA(\%)', 'mIOUs','mDICES']):
    #for idmet, met in enumerate(['Te. Loss','teOA(\%)', 'teAA(\%)', 'temIOUs','temDICES']):
    for idmet, met in enumerate(['Loss','OA(\%)', 'AA(\%)', 'mIOUs','mDICES']):
        for (idmodel, model), namelegend in zip(enumerate(models), ["Segnet", "Prop.-Segnet"]):
            avg = data_avg[idmodel,idmet]
            std = data_std[idmodel,idmet]
            if(met == 'Loss'):
                plt.plot(avg, label= namelegend)
            else:
                plt.plot(avg, label= met + " " + namelegend)
            plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
            
        if idmet == 0:
            plt.ylim(0.00,1.85)
        #plt.title("Training and Validation "+ met)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel(met, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        #plt.legend()
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.,prop={'size': 10})
        plt.savefig(met.replace("(\%)", "") + "SEGNET.png", bbox_inches='tight', pad_inches=.1)
        #plt.show()
        plt.clf()
