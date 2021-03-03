#!/bin/sh
# 
# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     python -u train.py --network_arch unet --use_augs --network_weights_path ./savedmodels/Unet_normal.pt --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_normal.pt --idtest $i
# 
#     python -u train.py --network_arch unet --use_augs --boxdown --network_weights_path ./savedmodels/Unet_normal_boxconv.pt --idtest $i
#     python -u test.py --network_arch unet --boxdown --network_weights_path ./savedmodels/Unet_normal_boxconv.pt --idtest $i
# 
# 
#     python -u train.py --network_arch unet --use_augs --use_SE --use_preluSE --network_weights_path ./savedmodels/Unet__SE_PRELU.pt --idtest $i
#     python -u test.py --network_arch unet  --use_SE --use_preluSE --network_weights_path ./savedmodels/Unet__SE_PRELU.pt --idtest $i
# 
# 
#     python -u train.py --network_arch unet --use_augs --use_SE --use_preluSE --boxdown --network_weights_path ./savedmodels/Unet_SE_PRELU_boxconv.pt --idtest $i
#     python -u test.py --network_arch unet  --use_SE --use_preluSE --boxdown --network_weights_path ./savedmodels/Unet_SE_PRELU_boxconv.pt --idtest $i
# 
# 
#     python -u train.py --network_arch unet --use_mini --use_augs --network_weights_path ./savedmodels/Unet_mini.pt --idtest $i
#     python -u test.py --network_arch unet --use_mini --network_weights_path ./savedmodels/Unet_mini.pt --idtest $i
# 
# 
#     python -u train.py --network_arch unet --use_mini --use_augs --boxdown --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --idtest $i
#     python -u test.py --network_arch unet --use_mini  --boxdown --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --idtest $i
# 
# 
#     python -u train.py --network_arch unet --use_mini --use_augs --use_SE --use_preluSE --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --idtest $i
#     python -u test.py --network_arch unet --use_mini --use_SE --use_preluSE --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --idtest $i
# 
# 
#     python -u train.py --network_arch unet --use_mini --use_augs --use_SE --use_preluSE --boxdown --network_weights_path ./savedmodels/Unet_mini_SE_PRELU_boxconv.pt --idtest $i
#     python -u test.py --network_arch unet --use_mini --use_SE --use_preluSE --boxdown --network_weights_path ./savedmodels/Unet_mini_SE_PRELU_boxconv.pt --idtest $i
# done


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

# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     python -u train.py --network_arch unet --use_augs --use_SE --network_weights_path ./savedmodels/Unet_normal_OnlySE.pt --idtest $i
#     python -u test.py --network_arch unet  --use_SE --network_weights_path ./savedmodels/Unet_normal_OnlySE.pt --idtest $i
#     
#     
#     python -u train.py --network_arch unet --use_augs --use_SE --boxdown --network_weights_path ./savedmodels/Unet_normal_OnlySE_boxconv.pt --idtest $i
#     python -u test.py --network_arch unet  --use_SE --boxdown --network_weights_path ./savedmodels/Unet_normal_OnlySE_boxconv.pt --idtest $i
#     
#     
#     from box_convolution import BoxConv2d
#     python -u train.py --network_arch unet --use_mini --use_augs --use_SE --network_weights_path ./savedmodels/Unet_mini_OnlySE.pt --idtest $i
#     python -u test.py --network_arch unet --use_mini --use_SE --network_weights_path ./savedmodels/Unet_mini_OnlySE.pt --idtest $i
#     
#     python -u train.py --network_arch unet --use_mini --use_augs --use_SE --boxdown --network_weights_path ./savedmodels/Unet_mini_OnlySE_boxdown.pt --idtest $i
#     python -u test.py --network_arch unet --use_mini --use_SE --boxdown --network_weights_path ./savedmodels/Unet_mini_OnlySE_boxdown.pt --idtest $i
#     
# done

# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     python -u train.py --network_arch resnet  --network_weights_path ./savedmodels/resnet_normal.pt --idtest $i
#     python -u test.py --network_arch resnet   --network_weights_path ./savedmodels/resnet_normal.pt --idtest $i
#     
#     
#     python -u train.py --network_arch resnet --boxdown  --network_weights_path ./savedmodels/resnet_boxconv.pt --idtest $i
#     python -u test.py --network_arch resnet --boxdown   --network_weights_path ./savedmodels/resnet_boxconv.pt --idtest $i
#     
# done

# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     python -u train.py --network_arch unet  --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet   --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch unet  --network_weights_path ./savedmodels/unet.pt --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet   --network_weights_path ./savedmodels/unet.pt --feature_scale 2 --idtest $i
#     
#     python -u train.py --network_arch unet  --network_weights_path ./savedmodels/unet.pt --feature_scale 4 --idtest $i
#     python -u test.py --network_arch unet   --network_weights_path ./savedmodels/unet.pt --feature_scale 4 --idtest $i
#     
#     
# done
# 
# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     
#     python -u train.py --network_arch unet --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet  --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 1 --idtest $i
#     
#     
#     python -u train.py --network_arch unet --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet  --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 2 --idtest $i
#     
#     
#     python -u train.py --network_arch unet --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 4 --idtest $i
#     python -u test.py --network_arch unet  --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 4 --idtest $i
# 
# done

# 
# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     python -u train.py --network_arch unet  --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet   --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch unet  --network_weights_path ./savedmodels/unet.pt --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet   --network_weights_path ./savedmodels/unet.pt --feature_scale 2 --idtest $i
#     
#     python -u train.py --network_arch unet  --network_weights_path ./savedmodels/unet.pt --feature_scale 4 --idtest $i
#     python -u test.py --network_arch unet   --network_weights_path ./savedmodels/unet.pt --feature_scale 4 --idtest $i
#     
#     
#     python -u train.py --network_arch unet --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet  --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 1 --idtest $i
#     
#     
#     python -u train.py --network_arch unet --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet  --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 2 --idtest $i
#     
#     
#     python -u train.py --network_arch unet --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 4 --idtest $i
#     python -u test.py --network_arch unet  --use_boxconv  --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 4 --idtest $i
# 
# done

# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --idtest $i
#     python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --idtest $i
#    
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 2 --idtest $i
#     
#     python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 2 --idtest $i
#     
#     python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs15.pt --feature_scale 1.5 --idtest $i
#     python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs15.pt --feature_scale 1.5 --idtest $i
#    
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 4 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 4 --idtest $i
#     
#     python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 4 --idtest $i
#     python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 4 --idtest $i
#     
#     python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs38.pt --feature_scale 3.8 --idtest $i
#     python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs38.pt --feature_scale 3.8 --idtest $i
#     
# 
# done
# 
# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     python -u train.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 1 --idtest $i
#     python -u test.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 1 --idtest $i
#     python -u test.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 0.75 --idtest $i
#     python -u test.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 0.75 --idtest $i
#     
#     
#     python -u train.py --use_mini --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 1 --idtest $i
#     python -u test.py --use_mini --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 1 --idtest $i
#     python -u test.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 0.76 --idtest $i
#     python -u test.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 0.76 --idtest $i
#     
#     python -u train.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 2 --idtest $i
#     python -u test.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 2 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 2 --idtest $i
#     python -u test.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 2 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 1.56 --idtest $i
#     python -u test.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 1.56 --idtest $i
#     
#     
#     python -u train.py --use_mini --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 2 --idtest $i
#     python -u test.py --use_mini --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 2 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 2 --idtest $i
#     python -u test.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 2 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 1.55 --idtest $i
#     python -u test.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 1.55 --idtest $i
#     
#     python -u train.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 4 --idtest $i
#     python -u test.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 4 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 4 --idtest $i
#     python -u test.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 4 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 2.6 --idtest $i
#     python -u test.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 2.6 --idtest $i
#     
#     python -u train.py --use_mini --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 4 --idtest $i
#     python -u test.py --use_mini --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 4 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 4 --idtest $i
#     python -u test.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 4 --idtest $i
#     
#     python -u train.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 3 --idtest $i
#     python -u test.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 3 --idtest $i
#     
# done

 rm ./output_test.txt
for i in 0 1 2 3 4
do
    rm ./savedmodels/*
    python -u train.py  --network_arch segnet --network_weights_path ./savedmodels/segnet.pt --idtest $i 
    python -u test.py   --network_arch segnet --network_weights_path ./savedmodels/segnet.pt --idtest $i 
    
    python -u train.py  --network_arch segnet --network_weights_path ./savedmodels/segnet_bx.pt --idtest $i --use_boxconv 
    python -u test.py   --network_arch segnet --network_weights_path ./savedmodels/segnet_bx.pt --idtest $i --use_boxconv

done
    




for i in 0 1 2 3 4
do 
    rm ./savedmodels/*
    python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
    python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
    
    python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --use_boxconv --idtest $i
    python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --use_boxconv --idtest $i
    
    python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --use_boxconv --idtest $i
    python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --use_boxconv --idtest $i
    
done
 
