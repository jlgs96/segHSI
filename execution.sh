#!/bin/sh

# for i in 0 1 2 3 4
for i in 3 4
do
    rm ./savedmodels/*
    python -u train.py --network_arch unet --use_augs --network_weights_path ./savedmodels/Unet_normal.pt --idtest $i
    python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_normal.pt --idtest $i

    python -u train.py --network_arch unet --use_augs --boxdown --network_weights_path ./savedmodels/Unet_normal_boxconv.pt --idtest $i
    python -u test.py --network_arch unet --boxdown --network_weights_path ./savedmodels/Unet_normal_boxconv.pt --idtest $i


    python -u train.py --network_arch unet --use_augs --use_SE --use_preluSE --network_weights_path ./savedmodels/Unet__SE_PRELU.pt --idtest $i
    python -u test.py --network_arch unet  --use_SE --use_preluSE --network_weights_path ./savedmodels/Unet__SE_PRELU.pt --idtest $i


    python -u train.py --network_arch unet --use_augs --use_SE --use_preluSE --boxdown --network_weights_path ./savedmodels/Unet_SE_PRELU_boxconv.pt --idtest $i
    python -u test.py --network_arch unet  --use_SE --use_preluSE --boxdown --network_weights_path ./savedmodels/Unet_SE_PRELU_boxconv.pt --idtest $i


    python -u train.py --network_arch unet --use_mini --use_augs --network_weights_path ./savedmodels/Unet_mini.pt --idtest $i
    python -u test.py --network_arch unet --use_mini --network_weights_path ./savedmodels/Unet_mini.pt --idtest $i


    python -u train.py --network_arch unet --use_mini --use_augs --boxdown --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --idtest $i
    python -u test.py --network_arch unet --use_mini  --boxdown --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --idtest $i


    python -u train.py --network_arch unet --use_mini --use_augs --use_SE --use_preluSE --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --idtest $i
    python -u test.py --network_arch unet --use_mini --use_SE --use_preluSE --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --idtest $i


    python -u train.py --network_arch unet --use_mini --use_augs --use_SE --use_preluSE --boxdown --network_weights_path ./savedmodels/Unet_mini_SE_PRELU_boxconv.pt --idtest $i
    python -u test.py --network_arch unet --use_mini --use_SE --use_preluSE --boxdown --network_weights_path ./savedmodels/Unet_mini_SE_PRELU_boxconv.pt --idtest $i
done


# for i in 0 1 2 3 4
# do
#     rm ./savedmodels/*
#     python -u train.py --network_arch unet --use_augs --network_weights_path ./savedmodels/Unet_normal.pt --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_normal.pt --idtest $i
# 
#     python -u train.py --network_arch unet --use_augs --boxdown --network_weights_path ./savedmodels/Unet_normal_boxconv.pt --idtest $i
#     python -u test.py --network_arch unet --boxdown --network_weights_path ./savedmodels/Unet_normal_boxconv.pt --idtest $i
# done
# 
