import numpy as np
import matplotlib.pyplot as plt
import argparse
if __name__ == "__main__":
    
    ##PINTAR DIAGRAMA DE BARRAS PLT.BAR Y EN EJE X: FEATURESCALES Y EN EJE Y: miouMAX
    models   = ['segnet', 'segnet_BOXCONV']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices','IOUs']
    #metrics = ['train','validation', 'oas', 'mpcas', 'mIOUs','dices']
    temetrics = ['test_losses', 'teoas', 'tempcas', 'temIOUs','tedices']
    valmetrics = ['val_losses', 'valoas', 'valmpcas', 'valmIOUs','valdices']
    metricsgraphs = ['Loss', 'OA(\%)', 'MPCA', 'mIOUs','eDice']
    params = [29474118,24821446]

    
    
    
    n_epochs = 60
    #n_seeds = 5
    n_seeds = 5
    
    data = np.ones((len(models), n_seeds, len(valmetrics), n_epochs)) * -1000.0
    for model in models:
        for seed in range(5):
            npzFile= np.load("./NPZs/Val/" + model +  '_' + str(seed)+'_'+ 'VAL' + '.npz')
            idmodel = models.index(model)
            #for idmet, met in enumerate(metrics):
            data[idmodel, seed, :, :] = npzFile['valData']

    pos_max = np.argmax(data[:,:,3,:],axis = 2)
    


    for model in models:
        for seed in range(5):
            npzFile= np.load("./NPZs/Test/"+ model + '_' + str(seed)+'_'+ 'TE' + '.npz')
            idmodel = models.index(model)
            #for idmet, met in enumerate(metrics):
            data[idmodel, seed, :, :] = npzFile['teData']



    data_max = np.ones((len(models), n_seeds, len(valmetrics))) * -1000.0
    #for idmet, met in enumerate(['Te. Loss','teOA(\%)', 'teAA(\%)', 'temIOUs','temDICES']):
    for (idmodel, model), namelegend in zip(enumerate(models), ["SEGNET", "SEGNETBX"]):
        for seed in range(5):
            data_max[idmodel, seed, :] = data[idmodel, seed, :, pos_max[idmodel, seed]]
    #print(data_max.shape)
    #exit()
    data_avg = np.average(data_max, axis=2)
    data_std = np.std(data_max, axis=2)
    #print(data_avg.shape)
    #exit()
            
    
    ####MODIFICAR ESTO PARA PODER RECORRER POR FEATURESCALES SOLO 
    #for idmet, met in enumerate(temetrics):
        #if idmet == 0: continue
        #space = -0.25
        #for (idmodel, model), namelegend, micolor in zip(enumerate(models), ["SEGNET", "SEGNETBX"], ['r', 'b']):
            #avg = data_avg[idmodel, :][idmet]    
            #std = data_std[idmodel, :][idmet]
            #plt.bar(float(1)+space, avg, width=0.25, color=micolor, label=model)
            #space += 0.25
        ##plt.ylim(0.75,1)
        #plt.xlim(0,2.25)
        ##plt.xticks([0.5-0.25/2, 1-0.25/2, 2-0.25/2], listfs)
        ##plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
        #plt.xlabel("Feature scale")
        #plt.ylabel(metricsgraphs[idmet])
        #plt.legend()
        #plt.show()


    ##MODIFICAR ESTO PARA PODER RECORRER POR FEATURESCALES SOLO
    names = ["SeNet", "Proposed-SeNet"]
    for idmet, met in enumerate(temetrics):
        if met != "temIOUs": continue
        if idmet == 0: continue
        for (idmodel, model), namelegend, micolor in zip(enumerate(models), ["SEGNET", "SEGNETBX"], ['r', 'b']):
            param = np.array(params[idmodel])//1000000
            avg = data_avg[idmodel, :][idmet]    
            std = data_std[idmodel, :][idmet]
            print(param, avg)
            plt.bar(param, avg, width=0.5, color=micolor, label=names[idmodel], yerr=std, align='center', ecolor='black', capsize=4)
        #plt.plot(np.array(params[:3])//1000000, data_avg[0, :, :][:,idmet], '--', c='r')
        #plt.plot(np.array(params[3:])//1000000, data_avg[1, :, :][:,idmet], '--', c='b')
        plt.xlabel("Parameters (Millions)", fontsize=15)
        plt.ylabel(metricsgraphs[idmet], fontsize=15)
        plt.legend()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        x1,x2,y1,y2 = plt.axis()
        #plt.axis((x1,x2,0.5,y2))
        plt.savefig("mIOU_SEGNET.png", bbox_inches='tight', pad_inches=.1)
        #plt.show()
