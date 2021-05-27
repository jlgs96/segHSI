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


for i in 0
do
    for j in 1
    do
        python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 
        python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j 
    done

    for j in 2 4 6 8 9 12
    do
        rm ./savedmodels/*
        
        python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 --use_head_box
        python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j --use_head_box
        
        python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 
        python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j 
    done
done


for i in 1 2 3 4 5 6 7 8 9
do
    for j in 1 2 4 6 8 9 12
    do
        rm ./savedmodels/*

        python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 --use_head_box
        python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j --use_head_box
        
        python -u train.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt   --use_boxconv --use_augs --idtest $i --resnet_blocks $j --batch-size 60 
        python -u test.py --network_arch resnet --network_weights_path ./savedmodels/resnet.pt --use_boxconv --idtest $i --resnet_blocks $j 
    done
done





    #     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 4   --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt --feature_scale 4 --idtest $i
#  
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_boxconv.pt   --use_boxconv --use_augs --feature_scale 4   --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_boxconv.pt --use_boxconv  --feature_scale 4 --idtest $i
#     
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 8   --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt --feature_scale 8 --idtest $i
#  
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_boxconv.pt   --use_boxconv --use_augs --feature_scale 8   --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_boxconv.pt --use_boxconv  --feature_scale 8 --idtest $i
#     
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.5   --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt --feature_scale 0.5 --idtest $i
#  
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_boxconv.pt   --use_boxconv --use_augs --feature_scale 0.5   --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_boxconv.pt --use_boxconv  --feature_scale 0.5 --idtest $i
#     
#     
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt   --use_augs --feature_scale 0.25   --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet.pt --feature_scale 0.25 --idtest $i
#  
#     python -u train.py --network_arch unet --network_weights_path ./savedmodels/Unet_boxconv.pt   --use_boxconv --use_augs --feature_scale 0.25   --idtest $i
#     python -u test.py --network_arch unet --network_weights_path ./savedmodels/Unet_boxconv.pt --use_boxconv  --feature_scale 0.25 --idtest $i
#     

#  done
