import numpy as np


#modelo,mini?,selayer?,prelu?,boxconv?
#unet,False,False,False,False,92.98394702320113,79.26230520986593,63.41824606688788,73.54459618340114


models = [ "SegNet", "SegNet_BOXCONV"]
#resnet_blocks = [1, 2, 4, 6, 8, 9, 12]
#Unet,False,False,False,False,0,4.0,92.113925528625,80.46037810036131,64.13218024295999,75.16298864465766
metrics = ["OA", "MPCA", "MIOU", "DICE"]

#features_unet          = [1,2,4]
#features_unetboxconv   = [1,2,4]

#features_unetmini          = [1,2,4]
#features_unetminiboxconv   = [1,2,4]

#features_unetminise          = [1,2,4]
#features_unetminiseboxconv   = [1,2,4]

n_runs=10
#values_all = np.ones((len(models), n_runs, len(resnet_blocks), len(metrics))) * -1000.0
values_all = np.ones((len(models),n_runs, len(metrics))) * -1000.0


for line in open("pruebaSegnet.txt"):
    if line:
        if len(line.strip()) <1: continue
        values  = line.strip().split(",")
        vals = np.array([float(a) for a in values[-4:]])
        try:
            idtest = int(values[5])
        except:
            print(line)
            exit()
        values[1:5] = [True if a == "True" else False for a in values[1:5]]

        #idresnetblock = resnet_blocks.index(int(values[6]))
        #if idtest > 7:
            #continue

        #print(values)
        if values[0] == "Unet":
            if not values[1] and not values[2] and not values[3] and not values[4]: mname = "Unet"
            elif not values[1] and not values[2] and not values[3] and values[4]:   mname = "Unet_BOXCONV"

            elif not values[1] and values[2] and values[3] and not values[4]:       mname = "Unet_PReLU"
            elif not values[1] and values[2] and values[3] and values[4]:           mname = "Unet_PReLU_BOXCONV"
            
            elif not values[1] and values[2] and not values[3] and not values[4]:   mname = "Unet_ReLU"
            elif not values[1] and values[2] and not values[3] and values[4]:       mname = "Unet_ReLU_BOXCONV"
                
            
            elif values[1] and not values[2] and not values[3] and not values[4]:   mname = "Unet-mini"
            elif values[1] and not values[2] and not values[3] and values[4]:       mname = "Unet-mini_BOXCONV"

            elif values[1] and values[2] and values[3] and not values[4]:           mname = "Unet-mini_PReLU"
            elif values[1] and values[2] and values[3] and values[4]:               mname = "Unet-mini_PReLU_BOXCONV"
            else:
                print("RED NO ESTÁ")
        elif values[0] == "Resnet":
            if not values[1] and not values[2] and not values[3] and not values[4]: mname = "ResNet"
            elif not values[1] and not values[2] and not values[3] and values[4]: mname = "ResNet_1BOXCONV"
            elif not values[1] and not values[2] and values[3] and values[4]: mname = "ResNet_2BOXCONV"
            else:
                print("RED NO ESTÁ")
        elif values[0] == "Segnet":
             if not values[1] and not values[2] and not values[3] and not values[4]: mname = "SegNet"
             elif not values[1] and not values[2] and not values[3] and values[4]: mname = "SegNet_BOXCONV"
             else:
                 print("RED NO ESTÁ")
        elif values[0] == "Enet":
             if not values[1] and not values[2] and not values[3] and not values[4]: mname = "ENet"
             else:
                 print("RED NO ESTÁ")
        elif values[0] == "BoxEnet":
            if not values[1] and not values[2] and not values[3] and not values[4]: mname = "BoxENet"
            else:
                print("RED NO ESTA")
        elif values[0] == "BoxOnlyENet":
            if not values[1] and not values[2] and not values[3] and not values[4]: mname = "BoxOnlyENet"
            else:
                print("RED NO ESTA")
        idmodel = models.index(mname)
        values_all[idmodel,idtest] = vals

values_all = values_all[:,:3]


values_all_ = np.sort(values_all , axis = 1)
#values_all_avg = np.average(values_all_[:,range(1,9),:], axis=1)
#values_all_std = np.std(values_all_[:,range(1,9),:], axis=1)
values_all_avg = np.average(values_all_, axis=1)
values_all_std = np.std(values_all_, axis=1)





for idmodel, model in enumerate(models):
    a = np.round(values_all_avg[idmodel,:], 2)
    b = np.round(values_all_std[idmodel,:], 2)
    string = model + " & "
    for idmet, met in enumerate(metrics):
        string += str(a[idmet])+"$\pm$"+str(b[idmet]) + " & "
    print(string[:-2] + r"\\")


#for idfeat, feat in enumerate(feature_scales):
    #print("-----", feat, "----------")
    #for idmodel, model in enumerate(models):
        #if idmodel > 1: continue
        #a = np.round(values_all_avg[idmodel, idfeat,:], 2)
        #b = np.round(values_all_std[idmodel, idfeat,:], 2)
        #string = model + " & "
        #for idmet, met in enumerate(metrics):
            #string += str(a[idmet])+"$\pm$"+str(b[idmet]) + " & "
        #print(string[:-2] + r"\\")
    #print("-------------------------")
    #print("-------------------------")

#for idresnetblock, resnetblock in enumerate(resnet_blocks):
    #print("-----", resnetblock, "----------")
    #for idmodel, model in enumerate(models):
        ##if idresnetblock > 1: continue
        #a = np.round(values_all_avg[idmodel, idresnetblock,:], 2)
        #b = np.round(values_all_std[idmodel, idresnetblock,:], 2)
        #string = model + " & "
        #for idmet, met in enumerate(metrics):
            #string += str(a[idmet])+"$\pm$"+str(b[idmet]) + " & "
        #print(string[:-2] + r"\\")
    #print("-------------------------")
    #print("-------------------------")










import matplotlib.pyplot as plt

pos = 0
for idmodel, model in enumerate(models):
    #if idmodel > 1: continue
    plt.bar(np.arange(len(values_all_avg[idmodel, :,2]))+pos, values_all_avg[idmodel, :,2], label=model, width=0.5)
    pos += 0.5
plt.xticks(np.arange(len(values_all_avg[idmodel, :,2]))+0.25, resnet_blocks)
plt.legend()
#plt.plot()
plt.show()




#paramsUNET   = list(reversed([490310, 1950086, 7778054, 31067654, 124181510, 496547846]))
#paramsUNETbx = list(reversed([271270, 1073734, 4272262, 17043718, 68084230, 272155654]))




paramsRESNET   = list([2098630, 3279302, 5640646, 8001990, 10363334, 11544006, 15086022])
paramsRESNETbx = list([954054, 990150, 1062342, 1134534, 1206726, 1242822, 1351110])
paramsRESNETbx2 = list([1526342, 2134726, 3351494, 4568262, 5785030, 6393414, 8218566])


params = [paramsRESNET, paramsRESNETbx, paramsRESNETbx2]

pos = 0
for idmodel, model in enumerate(models):
    ##if idmodel > 1: continue
    print(values_all_avg[idmodel, :,2])
    plt.plot(params[idmodel], values_all_avg[idmodel, :,2], label=model)#, width=0.5)
    #plt.plot(params[idmodel])
    #plt.plot(values_all_avg[idmodel, :,2], label=model)#, width=0.5)
    pos += 0.5
#plt.xticks(np.arange(len(values_all_avg[idmodel, :,2]))+0.25, feature_scales)
plt.legend()
#plt.plot()
plt.show()















#\begin{table}[!t]
#\let\center\empty
#\let\endcenter\relax
#\centering
#\caption{Resultados.}
#\multicolumn{5}{|c|}{3D conditional discriminator} \\
#\hline
#\textbf{Model} & \textbf{OA(\%)} & \textbf{MPCA} & \textbf{MIOU} & \textbf{DICE} \\
#\hline
#Unet & 93.31$\pm$0.17 & 80.52$\pm$3.01 & 67.76$\pm$3.47 & 78.11$\pm$3.62 \\
#Unet_BOXCONV & 93.57$\pm$1.13 & 83.93$\pm$2.16 & 72.91$\pm$2.81 & 82.83$\pm$2.28 \\
#Unet_PReLU & 93.39$\pm$0.32 & 82.38$\pm$2.58 & 66.49$\pm$2.68 & 76.91$\pm$2.96 \\
#Unet_PReLU_BOXCONV & 94.16$\pm$0.12 & 84.6$\pm$3.49 & 72.78$\pm$3.47 & 82.51$\pm$3.2 \\
#Unet\-mini & 93.58$\pm$0.14 & 77.76$\pm$2.65 & 65.17$\pm$2.29 & 75.07$\pm$2.61 \\
#Unet\-mini_BOXCONV & 93.51$\pm$0.26 & 85.6$\pm$2.17 & 68.3$\pm$2.72 & 78.71$\pm$2.75 \\
#Unet\-mini_PReLU & 94.08$\pm$0.05 & 88.15$\pm$0.36 & 73.1$\pm$0.93 & 83.2$\pm$0.76 \\
#Unet\-mini_PReLU_BOXCONV & 93.43$\pm$0.1 & 87.54$\pm$1.03 & 70.71$\pm$1.71 & 80.83$\pm$1.45 \\
#\hline
#\label{table:results}
#\end{table}
