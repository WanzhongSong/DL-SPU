# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import os
from path import Path
import random
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')


def image_path_city(root, name):
    return os.path.join(root, f'{name}')


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class teethmodel(Dataset):
    # datamode
    # 0--phaseW
    # 1--modu
    # 2--gray
    # 3--phaseW+modu
    # 4--phaseW+gray
    # 5--modu+gray
    # 6--phaseW+modu+gray
    # 7--pmi+intra_fn
    def __init__(self, root, transform=None,  subset="train", datamode=0, NUM_CHANNELS=3,filename='filename'):
        self.datamode = datamode
        self.root = Path(root)
        scene_list_path = self.root/subset+'.txt'
        self.scenes = [self.root/folder[:-1]
            for folder in open(scene_list_path)]
        sequence_set = []
        for scene in self.scenes:
            file_list = [name[:6]
                for name in open(Path(scene+"/"+filename+".txt"))]
            input_path = Path(scene+"/pmg_modu_new") #pmi
            if(datamode==7):
                input_path = Path(scene+"/pmg_modu_frame") #pmi with intra_frame_normal
            target_path = Path(scene+"/valid_mask_3C_disp_new")
            for i in range(len(file_list)):
                input_image = input_path/file_list[i]+'.png'
                traget_image = target_path/file_list[i]+'.png'
                sample = {'images': input_image, 'target': traget_image}
                sequence_set.append(sample)
        # random.shuffle(sequence_set)
        self.NUM_CHANNELS = NUM_CHANNELS
        self.samples = sequence_set
        self.transform = transform
    def __getitem__(self, index):
        datamode = self.datamode
        sample = self.samples[index]
        filename = sample['images']
        filenameGt = sample['target']
        image = load_image(filename)
        label = load_image(filenameGt)

        if self.transform is not None:
            image, label = self.transform(image, label)
       # phaseW
        if(datamode==0):
            image = image[[0]]
        # modu
        elif(datamode==1):
            image = image[[1]]
        # gray
        elif(datamode==2):
            image = image[[2]]
        # phaseW+modu
        elif(datamode==3):
            image = image[:2]
        # phaseW+gray
        elif(datamode==4):
            image = image[[0,2]]
        # modu+gray
        elif(datamode==5):
            image = image[1:]
        # phaseW+mou+gray
        elif(datamode==6 or datamode==7):
            image = image
        return image, label 
        # , filename, filenameGt

    def __len__(self):
        return len(self.samples)
