#!/bin/bash
chmod +x train.py
./train.py --network_arch ResNet --bands 10 --epochs 60

chmod +x test.py 
./test.py --network_arch resnet --bands 10
