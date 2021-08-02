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
    #print(data_max.shape)
    #exit()
    data_avg = np.average(data_max, axis=2)
    data_std = np.std(data_max, axis=2)
    #print(data_avg.shape)
    #exit()
            
    
    ####MODIFICAR ESTO PARA PODER RECORRER POR FEATURESCALES SOLO 
    #for idmet, met in enumerate(temetrics):
        #if idmet == 0: continue
        #for idfs, fs in enumerate(listfs):
            #space = -0.25
            #for (idmodel, model), namelegend, micolor in zip(enumerate(models), ["UNET", "UNETBX"], ['r', 'b']):
                #avg = data_avg[idmodel, idfs, :][idmet]    
                #std = data_std[idmodel, idfs, :][idmet]
                #if idfs != 0: plt.bar(float(fs)+space, avg, width=0.25, color=micolor)
                #else:         plt.bar(float(fs)+space, avg, width=0.25, color=micolor, label=model)
                #space += 0.25
        ##plt.ylim(0.75,1)
        #plt.xlim(0,2.25)
        #plt.xticks([0.5-0.25/2, 1-0.25/2, 2-0.25/2], listfs)
        ##plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=.1)
        #plt.xlabel("Feature scale")
        #plt.ylabel(metricsgraphs[idmet])
        ##plt.legend()
        ##plt.show()
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.,prop={'size': 10})
        ##percents = 100 * np.array(tr_percents[inicio:fin])
        ##plt.xticks(range(len(percents)), (np.array(percents)).astype("int"), fontsize=15)
        #plt.yticks(fontsize=15)
        #x1,x2,y1,y2 = plt.axis()
        #plt.axis((x1,x2,0.65,1))
        ##plt.savefig("OAvsPercent"+dset+".png", bbox_inches='tight', pad_inches=.1)
        #plt.show()


    ##MODIFICAR ESTO PARA PODER RECORRER POR FEATURESCALES SOLO
    names = ["UNet", "Proposed-UNet"]
    for idmet, met in enumerate(temetrics):
        if met != "temIOUs": continue
        if idmet == 0: continue
        for idfs, fs in enumerate(listfs):
            for (idmodel, model), namelegend, micolor in zip(enumerate(models), ["UNET", "UNETBX"], ['r', 'b']):
                param = np.array(params[idmodel*len(listfs)+idfs])//1000000
                avg = data_avg[idmodel, idfs, :][idmet]    
                std = data_std[idmodel, idfs, :][idmet]
                print(param, avg)
                if idfs != 0: plt.bar(param, avg, width=2, color=micolor, yerr=std, align='center', ecolor='black', capsize=4)
                else:         plt.bar(param, avg, width=2, color=micolor, label=names[idmodel], yerr=std, align='center', ecolor='black', capsize=4)
            plt.plot(np.array(params[:3])//1000000, data_avg[0, :, :][:,idmet], '--', c='r')
            plt.plot(np.array(params[3:])//1000000, data_avg[1, :, :][:,idmet], '--', c='b')
        plt.xlabel("Parameters (Millions)", fontsize=15)
        plt.ylabel(metricsgraphs[idmet], fontsize=15)
        plt.legend()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0.5,y2))
        plt.savefig("mIOU_UNET.png", bbox_inches='tight', pad_inches=.1)
        #plt.show()
