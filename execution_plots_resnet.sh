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
#                   python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnet.pt   --use_augs --resnet_blocks 6   --idtest 1 --npz_name resnet_RB_6_1
    python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnet.pt   --use_augs --resnet_blocks 6   --idtest $i --npz_name resnet_RB6
    python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnetbx.pt   --use_augs --resnet_blocks 6   --use_boxconv --idtest $i --npz_name resnet_BOXCONV_RB6
    
    python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnet.pt   --use_augs --resnet_blocks 9   --idtest $i --npz_name resnet_RB9
    python -u train_plot.py --network_arch resnet --network_weights_path ./savedmodels/Resnetbx.pt   --use_augs --resnet_blocks 9   --use_boxconv --idtest $i --npz_name resnet_BOXCONV_RB9
      
done
