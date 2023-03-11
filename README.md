# DL-SPU
Deep learning-enabled invalid-point removal for spatial phase unwrapping of 3D measurement

## Paper
Please click this (https://authors.elsevier.com/a/1gj%7EL6wNUutJV)

## Preamble
This code was developed and tested with python 3.6, Pytorch 1.8.0, and CUDA 10.2 on Ubuntu 18.04. It is based on [Eduardo Romera's ERFNet implementation (PyTorch Version)](https://github.com/Eromera/erfnet_pytorch). 

## Prerequisite
Please install the required packages: 
```
pip install -r requirement.txt
```

## Datasets
### Training set  
Please download them from [Part1](https://pan.baidu.com/s/1j9VzSCGPLqq8LyEhst_62g?pwd=tyrw), [Part2](https://pan.baidu.com/s/1nqYe3FavPxUW_GDr2pMvxQ?pwd=p67e), [Part3](https://pan.baidu.com/s/1Mmso2U2jCH2IuwNiR7cWcA?pwd=6u87) and [Part4](https://pan.baidu.com/s/1K-Y8hXEX2TS3nIQcRSfKDA?pwd=yxul).
### Validation set  
Please download it from [here](https://pan.baidu.com/s/1mEGDVNzrPY1YHOId_o4aVw?pwd=b86e)

## Training
Training the DL-SPU model from scratch by running
```bash
python train/train.py
```
Note: Please prepare the training datset before trainging.

## Pretrained model
Our pretrained model is in the file of "model_best.pth".

## Citation
```
@article{Luo2023,
author = {Luo, Xiaolong and Song, Wanzhong and Bai, Songlin and Li, Yu and Zhao, Zhihe},
journal = {Optics {\&} Laser Technology},
number = {March},
pages = {109340},
title = {{Deep learning-enabled invalid-point removal for spatial phase unwrapping of 3D measurement}},
volume = {163},
year = {2023}
}
```

## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License, which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
