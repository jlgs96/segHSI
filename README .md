# Efficient Segmentation of Hyperspectral Image using Adaptative Rectangular Convolution 
## Structure of the repository
This project is structured as it follows:
* [Aerial Data](/Aerial Data/) This directory contains the data that its needed(must run sampling_data.py first for fill the directories)
* [configs](/configs/) This directory contains a few examples for config files used in train and test functions.
* [helpers](/helpers/) This directory contains methods and functions very usefulls.
* [networks](/networks/) This directory contains the models and architectures of networks to work with.
* [savedmodels](/savedmodels/) This directory will contain all the saved models post training.

## Dataset

The scene images can be found [here](https://drive.google.com/drive/folders/1yCMqa9uDC_CEGtbnxeWEQCTb-odC2r4c?usp=sharing). The directory contains four files: 
1. image_rgb - The RGB rectified hyperspectral scene.
2. image_hsi_radiance - Radiance calibrated hyperspectral scene sampled at every 10th band (400nm, 410nm, 420nm, .. 900nm).
3. image_hsi_reflectance - Reflectance calibrated hyperspectral scene sampled at every 10th band.
4. image_labels - Semantic labels for the entire AeroCampus scene.

**this images must be downloaded before run sampling_data.py*


Note: The above files only contain every 10th band from 400nm to 900nm. You can request for the full size versions of both radiance and reflectance via an [email](mailto:aneesh.rangnekar@mail.rit.edu?subject=[GitHub]%20AeroCampus%20Full%20Version) to the corresponding author.


## Requirements

numpy 

cv2 (opencv-python)

pytorch1.0.1

Pillow

We recommend to use [Anaconda](https://www.anaconda.com/distribution/) environment for running all sets of code. We have tested our code on Ubuntu 18.04 with Python 3.7.

## Executing codes

1. Execute [samplin_data.py](/sampling_data.py/) to get train, validation and test splits with 64 x 64 image chips.
2. Execute [train.py](/train.py/) to start training and validation on the network you choose by parameters.
3. Make sure that the saved model from the training is on the directory.
4. Execute [test.py](/test.py/) to start the test on the saved model that you charge by parameters.

Before running any files, execute [sampling_data.py](/sampling_data.py/) to obtain train, validation and test splits with 64 x 64 image chips. 

Some of the important arguments used in [train](/train.py/) and [test](/test.py/) files are as follows:

Important Arguments

* config-file: Its the path to configuration file if is present.
* number of bands: how many bands you want to sample from HSI imagery (3-> RGB,4-> RGB + 1 Infrared, 6-> RGB + 3 Infrared, 10 and 31: Visible, 51: ALL, type = int.
* hsi_c: using radiance or reflectance for analysis.
* network_arch: choose which network architecture to use, by default ResNet.
* use_cuda: use or not use GPUs for processing

## License

This scene dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we (Rochester Institute of Technology) do not accept any responsibility for errors or omissions.
2. That you include a reference to the AeroCampus Dataset in any work that makes use of the dataset.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. You may not use the dataset or any derivative work for commercial purposes such as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (Rochester Institute of Technology).

## Citation
## TODO: PUT HERE OUR PAPER WHEN ITS DONE ##

When using the dataset or code, please cite our [paper](https://arxiv.org/pdf/1912.08178.pdf): 
```
@misc{rangnekar2019aerorit,
    title={AeroRIT: A New Scene for Hyperspectral Image Analysis},
    author={Aneesh Rangnekar and Nilay Mokashi and Emmett Ientilucci and Christopher Kanan and Matthew J. Hoffman},
    year={2019},
    eprint={1912.08178},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
## TODO: PUT HERE OUR PAPER WHEN ITS DONE ##
## Acknowledgements

The codebase is heavily based off [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) , [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg), [AeroRIT](https://github.com/aneesh3108/AeroRIT),[BoxConvolutions](https://github.com/shrubb/box-convolutions) . Both are great repositories - have a look!


