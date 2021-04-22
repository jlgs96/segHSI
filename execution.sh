#!/bin/sh
rm ./output_test.txt
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

# rm ./output_test.txt
# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet_9block.pt --resnet_blocks 9 --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet_9block.pt --resnet_blocks 9 --idtest $i
# 
#     python -u train.py --network_arch resnet --use_boxconv --network_weights_path ./savedmodels/resnet_9blockbx.pt --resnet_blocks 9 --idtest $i
#     python -u test.py --network_arch resnet --use_boxconv --network_weights_path ./savedmodels/resnet_9blockbx.pt --resnet_blocks 9 --idtest $i
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet_12block.pt --resnet_blocks 12 --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet_12block.pt --resnet_blocks 12 --idtest $i
# 
#     python -u train.py --network_arch resnet --use_boxconv --network_weights_path ./savedmodels/resnet_12blockbx.pt --resnet_blocks 12 --idtest $i
#     python -u test.py --network_arch resnet --use_boxconv --network_weights_path ./savedmodels/resnet_12blockbx.pt --resnet_blocks 12 --idtest $i
# 
#     python -u train.py --network_arch resnet  --network_weights_path ./savedmodels/resnet_normal.pt --idtest $i
#     python -u test.py --network_arch resnet   --network_weights_path ./savedmodels/resnet_normal.pt --idtest $i
#     
#     python -u train.py --network_arch resnet --use_boxconv  --network_weights_path ./savedmodels/resnet_boxconv.pt --idtest $i
#     python -u test.py --network_arch resnet --use_boxconv   --network_weights_path ./savedmodels/resnet_boxconv.pt --idtest $i
#     
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

# for i in 0 1 2 3 4
# 
# do
#     rm ./savedmodels/*
#     
#     python -u train.py --network_arch unet   --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet    --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch unet   --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet    --network_weights_path ./savedmodels/unet_boxconv.pt --feature_scale 1 --idtest $i
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
# 
#  for i in 0 1 2 3 4
# # 
#  do
#      rm ./savedmodels/*
#      python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#      python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
# #     
# #      python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --idtest $i
# #      python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --idtest $i
# #     
#      python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --idtest $i
#      python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --idtest $i
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
#  done
# # 
#  for i in 0 1 2 3 4
# # 
#  do
#      rm ./savedmodels/*
#      python -u train.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 1 --idtest $i
#      python -u test.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 1 --idtest $i
# #     
# #      python -u train.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 1 --idtest $i
# #      python -u test.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 1 --idtest $i
# #     
#      python -u train.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 0.75 --idtest $i
#      python -u test.py --use_mini --use_boxconv --network_arch unet --network_weights_path ./savedmodels/Unet_mini_boxconv.pt --feature_scale 0.75 --idtest $i
# #     
# #     
#      python -u train.py --use_mini --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 1 --idtest $i
#      python -u test.py --use_mini --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 1 --idtest $i
# #     
# #    python -u train.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 1 --idtest $i
# #    python -u test.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 1 --idtest $i
# #     
#      python -u train.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 0.76 --idtest $i
#      python -u test.py --use_mini --use_boxconv --use_SE --use_preluSE --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv.pt --feature_scale 0.76 --idtest $i
# #     
# #     python -u train.py --use_mini --network_arch unet --network_weights_path ./savedmodels/unet_mini.pt --feature_scale 2 --idtest $i
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
#  done
# # 
# #  rm ./output_test.txt
# # for i in 0 1 2 3 4
#  for i in 0 1 2 3 4
# # 
#  do
#      rm ./savedmodels/*
#      python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#      python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
# #     
# #      python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --idtest $i
# #      python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --idtest $i
# #     
#      python -u train.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --idtest $i
#      python -u test.py --network_arch unet --use_boxconv --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --idtest $i
# do
#     rm ./savedmodels/*
#     python -u train.py  --network_arch segnet --network_weights_path ./savedmodels/segnet.pt --idtest $i 
#     python -u test.py   --network_arch segnet --network_weights_path ./savedmodels/segnet.pt --idtest $i 
#     
#     python -u train.py  --network_arch segnet --network_weights_path ./savedmodels/segnet_bx.pt --idtest $i --use_boxconv 
#     python -u test.py   --network_arch segnet --network_weights_path ./savedmodels/segnet_bx.pt --idtest $i --use_boxconv
# 
# done
    




# for i in 0 1 2 3 4
# do 
#     rm ./savedmodels/*
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --use_boxconv --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet_bx.pt --feature_scale 1 --use_boxconv --idtest $i
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --use_boxconv --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --use_boxconv --idtest $i
#     
# done
# for i in 0 1 2 3 4
# do
#     rm ./savedmodels/*
#     python -u train.py --network_arch enet --network_weights_path ./savedmodels/enet.pt --learning-rate 1e-3 --idtest $i
#     python -u test.py --network_arch enet --network_weights_path ./savedmodels/enet.pt  --idtest $i
#     
#     python -u train.py --network_arch enet --network_weights_path ./savedmodels/enet.pt --learning-rate 1e-3 --idtest $i --use_mini
#     python -u test.py --network_arch enet --network_weights_path ./savedmodels/enet.pt --idtest $i --use_mini
#     
#     python -u train.py --network_arch boxenet --network_weights_path ./savedmodels/boxenet.pt --learning-rate 1e-3 --idtest $i
#     python -u test.py --network_arch boxenet --network_weights_path ./savedmodels/boxenet.pt --idtest $i
#     
#     python -u train.py --network_arch boxenet --network_weights_path ./savedmodels/boxenet.pt --learning-rate 1e-3 --idtest $i --use_mini
#     python -u test.py --network_arch boxenet --network_weights_path ./savedmodels/boxenet.pt --idtest $i --use_mini
#     
#     python -u train.py --network_arch boxonlyenet --network_weights_path ./savedmodels/boxonlyenet.pt --learning-rate 1e-3 --idtest $i
#     python -u test.py --network_arch boxonlyenet --network_weights_path ./savedmodels/boxonlyenet.pt --idtest $i
#     
#     python -u train.py --network_arch boxonlyenet --network_weights_path ./savedmodels/boxonlyenet.pt --learning-rate 1e-3 --idtest $i --use_mini
#     python -u test.py --network_arch boxonlyenet --network_weights_path ./savedmodels/boxonlyenet.pt --idtest $i --use_mini
#     
#  
# done


# 
# for i in 0 1 2 3 4 5 6 7 8 9
# do 
#     rm ./savedmodels/*
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --use_augs --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet_bx_fs1.pt --use_augs --feature_scale 1 --use_boxconv --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet_bx_fs1.pt --feature_scale 1 --use_boxconv --idtest $i
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet_bx_fs075.pt --use_augs --feature_scale 0.75 --use_boxconv --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet_bx_fs075.pt --feature_scale 0.75 --use_boxconv --idtest $i
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --use_augs --feature_scale 1 --use_mini --use_SE --use_preluSE  --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu.pt --feature_scale 1 --use_mini --use_SE --use_preluSE --idtest $i
#     
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_bx.pt --use_boxconv --use_augs --feature_scale 1 --use_mini --use_SE --use_preluSE  --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_bx.pt --use_boxconv --feature_scale 1 --use_mini --use_SE --use_preluSE --idtest $i
#     
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv_fs.pt --use_augs --feature_scale 0.75 --use_mini --use_SE --use_preluSE --use_boxconv --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_mini_SE_Prelu_boxconv_fs.pt --feature_scale 0.75 --use_mini --use_SE --use_preluSE --use_boxconv --idtest $i
# done

# 
# for i in 0 1 2 3 4 5 6 7 8 9
# do 
#     rm ./savedmodels/*
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet6.pt --use_augs --feature_scale 1 --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet6.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet6_bx_fs1.pt --use_augs --feature_scale 1 --use_boxconv --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet6_bx_fs1.pt --feature_scale 1 --use_boxconv --idtest $i
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet_9block.pt --use_augs --feature_scale 1 --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet_9block.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet_9blockbx.pt --use_augs --feature_scale 1 --use_boxconv --idtest $i --resnet_blocks 9
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet_9blockbx.pt --feature_scale 1 --use_boxconv --idtest $i --resnet_blocks 9
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet_9block.pt --use_augs --feature_scale 1 --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet_9block.pt --feature_scale 1 --idtest $i
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet_18blockbx.pt --use_augs --feature_scale 1 --use_boxconv --idtest $i --resnet_blocks 18
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet_18blockbx.pt --feature_scale 1 --use_boxconv --idtest $i --resnet_blocks 18
#     
#     
#     
#     
#     
#     
# done

 for i in 0 1 2 3 4 5 6 7 8 9
 do 
    rm ./savedmodels/*
    
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet_bx.pt --learning-rate 1e-3 --use_boxconv --use_augs --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet_bx.pt --use_boxconv --feature_scale 2  --idtest $i
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --use_augs --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet.pt --feature_scale 2  --idtest $i
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/unet_bx.pt --use_boxconv --use_augs --feature_scale 2 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/unet_bx.pt --use_boxconv --feature_scale 2  --idtest $i



# 
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet6.pt --use_augs --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet6.pt --idtest $i
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet6bx.pt --use_boxconv --use_augs --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet6bx.pt --use_boxconv --idtest $i
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet6bxlr3.pt --use_boxconv --use_augs --learning-rate 1e-3 --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet6bxlr3.pt --use_boxconv --idtest $i
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet9.pt --resnet_blocks 9 --use_augs --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet9.pt --resnet_blocks 9 --idtest $i
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet9bx.pt --resnet_blocks 9 --use_boxconv --use_augs --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet9bx.pt --resnet_blocks 9 --use_boxconv --idtest $i
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet9lr3bx.pt --resnet_blocks 9 --use_boxconv --use_augs --learning-rate 1e-3 --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet9lr3bx.pt --resnet_blocks 9 --use_boxconv --idtest $i
#     
    
#     python -u train.py --network_arch segnet --network_weights_path ./savedmodels/segnet.pt --use_augs --idtest $i
#     python -u test.py --network_arch segnet --network_weights_path ./savedmodels/segnet.pt --idtest $i
#     
#     
#     python -u train.py --network_arch segnet --network_weights_path ./savedmodels/segnet_bx.pt --use_augs --use_boxconv --idtest $i
#     python -u test.py --network_arch segnet --network_weights_path ./savedmodels/segnet_bx.pt --use_boxconv --idtest $i
#     
#     python -u train.py --network_arch enet --network_weights_path ./savedmodels/enet.pt --learning-rate 1e-3 --use_augs --idtest $i
#     python -u test.py --network_arch enet --network_weights_path ./savedmodels/enet.pt  --idtest $i
#     
#     python -u train.py --network_arch boxenet --network_weights_path ./savedmodels/boxenet.pt --learning-rate 1e-3 --use_augs --idtest $i
#     python -u test.py --network_arch boxenet --network_weights_path ./savedmodels/boxenet.pt --idtest $i
#     
#     python -u train.py --network_arch boxonlyenet --network_weights_path ./savedmodels/boxonlyenet.pt --learning-rate 1e-3 --use_augs --idtest $i
#     python -u test.py --network_arch boxonlyenet --network_weights_path ./savedmodels/boxonlyenet.pt --idtest $i
#     
#       
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet18.pt --resnet_blocks 18 --use_augs --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet18.pt --resnet_blocks 18 --idtest $i
# 
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet18lr3bx.pt --resnet_blocks 18 --use_boxconv --use_augs --learning-rate 1e-3 --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet18lr3bx.pt --resnet_blocks 18 --use_boxconv --idtest $i
   
#      #UNET CON PRELU Y SE
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet__SE_PRELU.pt --use_SE --use_preluSE --use_augs --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet__SE_PRELU.pt --use_SE --use_preluSE --feature_scale 1   --idtest $i
#     
    python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_SE_PRELU_boxconv.pt --learning-rate 1e-3 --use_SE --use_preluSE --use_augs --feature_scale 1 --use_boxconv  --idtest $i
    python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_SE_PRELU_boxconv.pt --use_SE --use_preluSE  --feature_scale 1 --use_boxconv  --idtest $i
    
    #UNET CON PRELU Y SE A FS MODIFICADO
    python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_SE_PRELU_boxconv_fs.pt   --use_SE --use_preluSE --use_augs --feature_scale 0.76 --use_boxconv  --idtest $i
    python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_SE_PRELU_boxconv_fs.pt --use_SE --use_preluSE --feature_scale 0.76 --use_boxconv  --idtest $i
    
    
    #UNET CON RELU Y SE
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet__SE_RELU.pt --use_SE --use_augs --feature_scale 1 --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet__SE_RELU.pt --use_SE --feature_scale 1   --idtest $i
#     
    python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_SE_RELU_boxconv.pt --learning-rate 1e-3 --use_SE --use_augs --feature_scale 1 --use_boxconv  --idtest $i
    python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_SE_RELU_boxconv.pt --use_SE   --feature_scale 1 --use_boxconv  --idtest $i

    #UNET CON RELU Y SE A FS MODIFICADO
    python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_SE_RELU_boxconv_fs.pt --learning-rate 1e-3  --use_SE  --use_augs --feature_scale 0.76 --use_boxconv  --idtest $i
    python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_SE_RELU_boxconv_fs.pt --use_SE --feature_scale 0.76 --use_boxconv  --idtest $i

   done
