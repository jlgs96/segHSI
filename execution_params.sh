#!/bin/sh


for j in 1 2 4 6 8 9 12
do
    python -u get_params.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_head_box --use_augs --idtest 0 --resnet_blocks $j --batch-size 60
    python -u get_params.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest 0 --resnet_blocks $j --batch-size 60
done
