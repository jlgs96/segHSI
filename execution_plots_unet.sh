#!/bin/sh
# cd ./randomSampling
# for i in 6 7 8 9
# do
#   
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_augs --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --idtest $i
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_augs --idtest $i --resnet_blocks 9
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --idtest $i --resnet_blocks 9
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks 9
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks 9
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_augs --idtest $i --resnet_blocks 18
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --idtest $i --resnet_blocks 18
#     
#     
#     python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks 18
#     python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks 18
# 
# done
# 
# cd ..
    

#  for i in 0 1 2 3 4 5 6 7 8 9
#  do
#     for j in 1 2 4 6 8 9 12
#     do
#         rm ./savedmodels/*
# 
#         
#         
#         
#         python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 --use_head_box
#         python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j --use_head_box
#         
#         
#     
#         
#         python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 
#         python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j 
#         
#     done
# done

# 
# for i in 0
# do
#     for j in 1
#     do
#         python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 
#         python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j 
#     done
# 
#     for j in 2 4 6 8 9 12
#     do
#         rm ./savedmodels/*
#         
#         python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 --use_head_box
#         python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j --use_head_box
#         
#         python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 
#         python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j 
#     done
# done
# 
# 
# for i in 1 2 3 4 5 6 7 8 9
# do
#     for j in 1 2 4 6 8 9 12
#     do
#         rm ./savedmodels/*
# 
#         python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 --use_head_box
#         python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j --use_head_box
#         
#         python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 
#         python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j 
#     done
# done


for i in 0 1 2 3 4
do
#     python -u train.py --network_arch segnet --network_weights_path ./savedmodels/segnet.pt   --use_augs --idtest $i
#     python -u test.py --network_arch segnet --network_weights_path ./savedmodels/segnet.pt   --idtest $i
#         
#     python -u train.py --network_arch segnet --network_weights_path ./savedmodels/segnet.pt   --use_boxconv --use_augs --idtest $i
#     python -u test.py --network_arch segnet --network_weights_path ./savedmodels/segnet.pt   --use_boxconv  --idtest $i
    
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.5   --idtest $i --npz_name unet_fs0.5_$i
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 0.5 --use_boxconv   --idtest $i --npz_name unet_BOXCONV_fs0.5_$i
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.7   --idtest $i --npz_name unet_fs0.7_$i

    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 1   --idtest $i --npz_name unet_fs1_$i
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 1 --use_boxconv   --idtest $i --npz_name unet_BOXCONV_fs1_$i

    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 2   --idtest $i --npz_name unet_fs2_$i
    python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unetbx.pt   --use_augs --feature_scale 2 --use_boxconv   --idtest $i --npz_name unet_BOXCONV_fs2_$i

#     python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.7   --use_boxconv --idtest $i --npz_name unet_BOXCONV_fs1_$i
#     
#     python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.5   --idtest $i --npz_name unet_fs05_$i
#     python -u train_plot.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.5   --use_boxconv --idtest $i --npz_name unet_BOXCONV_fs05_$i
    

done
