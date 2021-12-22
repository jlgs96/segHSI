#!/bin/sh
for i in 0 1 2 3 4
do
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.5   --idtest $i --npz_name unet_fs0.5_$i
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 0.5 --use_boxconv   --idtest $i --npz_name unet_BOXCONV_fs0.5_$i

    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 1   --idtest $i --npz_name unet_fs1_$i
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 1 --use_boxconv   --idtest $i --npz_name unet_BOXCONV_fs1_$i

    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 2   --idtest $i --npz_name unet_fs2_$i 
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 2 --use_boxconv   --idtest $i --npz_name unet_BOXCONV_fs2_$i 
    
done
