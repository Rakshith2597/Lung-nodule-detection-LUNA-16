#Code written by Rakshith Sathish
#The work is made public with MIT License

import os
import collections
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


from torch.utils import data


class lunaLoader(data.Dataset):
    def __init__(self,split="train",is_transform=True,img_size=512):
        self.split = split
        self.path= "/home/rak/data/"+self.split
        self.is_transform = is_transform
        self.img_size = img_size
        self.files = os.listdir(self.path+'/images/') # [image1_img.npy, image2_img.npy]
        
        self.img_tf = transforms.Compose(
            [   transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([-460.466],[444.421])                
            ])
        
        self.label_tf = transforms.Compose(
            [   
            	transforms.Resize(self.img_size,interpolation=0),
                transforms.ToTensor(),
            ])
        
    
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,index):
        fname = self.files[index] # image1_img.npy, image1_label.npy
        img = Image.fromarray(np.load(self.path+'/images/'+fname).astype(float))
        im_id = fname.split('_')[1]
        label = Image.fromarray(np.load(self.path+'/labels_small/masks_'+im_id))
        
        if self.is_transform:
            img, label = self.transform(img,label)
        
        return img, label.squeeze(0)
    
    def transform(self,img,label):
        img = self.img_tf(img)
        label = self.label_tf(label)
        
        return img,label

