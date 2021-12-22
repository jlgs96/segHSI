#!/bin/sh
for i in 0 1 2 3 4
do
#                   
    python -u train_plot.py --network_arch segnet --network_weights_path ./savedmodels/Segnet.pt   --use_augs   --idtest $i --npz_name segnet
    python -u train_plot.py --network_arch segnet --network_weights_path ./savedmodels/SegnetBX.pt   --use_augs  --use_boxconv --idtest $i --npz_name segnet_BOXCONV
done
