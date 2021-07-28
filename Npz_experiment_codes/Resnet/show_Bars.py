import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
if __name__ == "__main__":
    
    ##PINTAR DIAGRAMA DE BARRAS PLT.BAR Y EN EJE X: FEATURESCALES Y EN EJE Y: miouMAX
    listfs   = ["0.5", "1", "2"]
    models   = ['unet', 'unet_BOXCONV']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices','IOUs']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices']
    temetrics = ['test_losses', 'teoas', 'tempcas', 'temIOUs','tedices']
    valmetrics = ['val_losses', 'valoas', 'valmpcas', 'valmIOUs','valdices']
    metricsgraphs = ['Loss', 'OA(\%)', 'MPCA', 'mIOUs','eDice']
    params = [124181510, 31067654, 7778054, 68084230, 17043718, 4272262]

    
    
    
    n_epochs = 60
    #n_seeds = 5
    n_seeds = 5
    data = np.ones((len(models), len(listfs), n_seeds, len(valmetrics), n_epochs)) * -1000.0

    for fs in listfs:
        for model in models:
            
            for seed in range(5):
                npzFile= np.load("/home/josel/Escritorio/Npz_experiment_codes/Unet/NPZs/Validation/" + model + '_fs' + str(fs) + '_' + str(seed)+'_'+ 'VAL' + '.npz')
                idfs    = listfs.index(fs)
                idmodel = models.index(model)
                #for idmet, met in enumerate(metrics):
                data[idmodel, idfs, seed, :, :] = npzFile['valData']

    pos_max = np.argmax(data[:,:,:,3,:],axis = 3)
    


    for fs in listfs:
        for model in models:
            for seed in range(5):
                npzFile= np.load("/home/josel/Escritorio/Npz_experiment_codes/Unet/NPZs/Test/"+ model + '_fs' + str(fs) + '_' + str(seed)+'_'+ 'TE' + '.npz')
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
    #print(data_max.shape)
    #exit()
    data_avg = np.average(data_max, axis=2)
    data_std = np.std(data_max, axis=2)
    #print(data_avg.shape)
    #exit()
            
    
    ###MODIFICAR ESTO PARA PODER RECORRER POR FEATURESCALES SOLO 
    for idmet, met in enumerate(temetrics):
        if idmet == 0: continue
        for idfs, fs in enumerate(listfs):
            space = -0.25
            for (idmodel, model), namelegend, micolor in zip(enumerate(models), ["UNET", "UNETBX"], ['r', 'b']):
                avg = data_avg[idmodel, idfs, :][idmet]    
                std = data_std[idmodel, idfs, :][idmet]
                if idfs != 0: plt.bar(float(fs)+space, avg, width=0.25, color=micolor)
                else:         plt.bar(float(fs)+space, avg, width=0.25, color=micolor, label=model)
                space += 0.25
        #plt.ylim(0.75,1)
        plt.xlim(0,2.25)
        plt.xticks([0.5-0.25/2, 1-0.25/2, 2-0.25/2], listfs)
        #plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
        plt.xlabel("Feature scale")
        plt.ylabel(metricsgraphs[idmet])
        plt.legend()
        plt.show()



    ##MODIFICAR ESTO PARA PODER RECORRER POR FEATURESCALES SOLO 
    for idmet, met in enumerate(temetrics):
        if idmet == 0: continue
        for idfs, fs in enumerate(listfs):
            #space = -0.25
            for (idmodel, model), namelegend, micolor in zip(enumerate(models), ["UNET", "UNETBX"], ['r', 'b']):
                #param = params[idfs*len(models)+idmodel]
                param = params[idmodel*len(listfs)+idfs]
                avg = data_avg[idmodel, idfs, :][idmet]    
                std = data_std[idmodel, idfs, :][idmet]
                print(param, avg)
                if idfs != 0: plt.bar(param, avg, width=5e6, color=micolor)
                else:         plt.bar(param, avg, width=5e6, color=micolor, label=model)
                #space += 0.25
            plt.plot(params[:3], data_avg[0, :, :][:,idmet], '--', c='r')
            plt.plot(params[3:], data_avg[1, :, :][:,idmet], '--', c='b')
        #plt.ylim(0.75,1)
        #plt.xlim(0,2.25)
        #plt.xticks([0.5-0.25/2, 1-0.25/2, 2-0.25/2], listfs)
        #plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
        plt.xlabel("Feature scale")
        plt.ylabel(metricsgraphs[idmet])
        plt.legend()
        plt.show()
