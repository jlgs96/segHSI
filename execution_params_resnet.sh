#!/bin/sh
for j in 6 9
do
    python -u get_params.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt    --use_augs --idtest 0 --resnet_blocks $j 
    python -u get_params.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv  --use_augs --idtest 0 --resnet_blocks $j 
done
