#!/bin/sh

for i in 0 1 2 3 4
do
    python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnet.pt   --use_augs --resnet_blocks 6   --idtest $i --npz_name resnet_RB6
    python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnetbx.pt   --use_augs --resnet_blocks 6   --use_boxconv --idtest $i --npz_name resnet_BOXCONV_RB6
    
    python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnet.pt   --use_augs --resnet_blocks 9   --idtest $i --npz_name resnet_RB9
    python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnetbx.pt   --use_augs --resnet_blocks 9   --use_boxconv --idtest $i --npz_name resnet_BOXCONV_RB9
      
done
