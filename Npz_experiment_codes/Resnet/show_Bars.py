import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
if __name__ == "__main__":
    
    ##PINTAR DIAGRAMA DE BARRAS PLT.BAR Y EN EJE X: FEATURESCALES Y EN EJE Y: miouMAX
    listrb   = ["6","9"]
    models   = ['resnet', 'resnet_BOXCONV']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices','IOUs']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices']
    temetrics = ['test_losses', 'teoas', 'tempcas', 'temIOUs','tedices']
    valmetrics = ['val_losses', 'valoas', 'valmpcas', 'valmIOUs','valdices']
    metricsgraphs = ['Loss', 'OA(\%)', 'MPCA', 'mIOUs','eDice']
    params = [8001990, 11544006, 4568262, 6393414]



    
    
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
    #print(data_max.shape)
    #exit()
    data_avg = np.average(data_max, axis=2)
    data_std = np.std(data_max, axis=2)
    #print(data_avg.shape)
    #exit()
            
    
    ###MODIFICAR ESTO PARA PODER RECORRER POR FEATURESCALES SOLO 
    for idmet, met in enumerate(temetrics):
        if idmet == 0: continue
        for idrb, rb in enumerate(listrb):
            space = -0.25
            for (idmodel, model), namelegend, micolor in zip(enumerate(models), ["RESNET", "RESNETBX"], ['r', 'b']):
                avg = data_avg[idmodel, idrb, :][idmet]    
                std = data_std[idmodel, idrb, :][idmet]
                if idrb != 0: plt.bar(float(rb)+space, avg, width=0.25, color=micolor)
                else:         plt.bar(float(rb)+space, avg, width=0.25, color=micolor, label=model)
                space += 0.25
        #plt.ylim(0.75,1)
        #plt.xlim(0,2.25)
        #plt.xticks([0.5-0.25/2, 1-0.25/2], listrb)
        #plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
        plt.xlabel("Resnet Blocks")
        plt.ylabel(metricsgraphs[idmet])
        plt.legend()
        plt.show()



    ##MODIFICAR ESTO PARA PODER RECORRER POR FEATURESCALES SOLO 
    for idmet, met in enumerate(temetrics):
        if idmet == 0: continue
        for idrb, rb in enumerate(listrb):
            #space = -0.25
            for (idmodel, model), namelegend, micolor in zip(enumerate(models), ["UNET", "UNETBX"], ['r', 'b']):
                #param = params[idrb*len(models)+idmodel]
                param = params[idmodel*len(listrb)+idrb]
                avg = data_avg[idmodel, idrb, :][idmet]    
                std = data_std[idmodel, idrb, :][idmet]
                print(param, avg)
                if idrb != 0: plt.bar(param, avg, width=5e5, color=micolor)
                else:         plt.bar(param, avg, width=5e5, color=micolor, label=model)
                #space += 0.25
                plt.plot(params[:2], data_avg[0, :, :][:,idmet], '--', c='r')
                plt.plot(params[2:], data_avg[1, :, :][:,idmet], '--', c='b')
            #plt.ylim(0.75,1)
            #plt.xlim(0,2.25)
            #plt.xticks([0.5-0.25/2, 1-0.25/2, 2-0.25/2], listrb)
            #plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
            plt.xlabel("Feature scale")
            plt.ylabel(metricsgraphs[idmet])
            plt.legend()
            plt.show()
