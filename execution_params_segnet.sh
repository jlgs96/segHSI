#!/bin/sh


    python -u get_params.py --network_arch segnet --network_weights_path ./savedmodels/segnet.pt    --use_augs --idtest 0 
    python -u get_params.py --network_arch segnet --network_weights_path ./savedmodels/segnetbx.pt   --use_boxconv  --use_augs --idtest 0 
