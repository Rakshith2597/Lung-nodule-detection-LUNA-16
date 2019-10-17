import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SUMNet_bn import SUMNet
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
   
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

luna_subset_path = '/home/siplab/rachana/rak/dataset/subset3/'
img_file = '/home/siplab/rachana/rak/dataset/subset3/1.3.6.1.4.1.14519.5.2.1.6279.6001.292057261351416339496913597985.mhd'
itk_img = sitk.ReadImage(img_file) 
img_tensor = torch.from_numpy(sitk.GetArrayFromImage(itk_img)).unsqueeze(1).float()
# normalization to [0,1]
img_tensor_norm = img_tensor-img_tensor.min()
img_tensor_norm = img_tensor_norm/img_tensor_norm.max()


seg_model_loadPath = '/home/siplab/rachana/rak/Results/SUMNet/Adam_1e-4_ep100/'
netS = SUMNet(in_ch=1,out_ch=2)
netS.load_state_dict(torch.load(seg_model_loadPath+'sumnet_cpu.pt'))

# netS = netS.cuda()
savePath = seg_model_loadPath+'seg_results/'
if not os.path.isdir(savePath):
    os.makedirs(savePath)

for sliceNum in range(img_tensor_norm.shape[0]):
    img_slice = img_tensor_norm[sliceNum].unsqueeze(0)
    out = F.softmax(netS(img_slice),dim=1)
    
    plt.figure(figsize=[15,5])
    plt.subplot(121)
    plt.imshow(img_slice.squeeze(0).squeeze(0).numpy(),cmap='gray')
    plt.title('Original Image')
    plt.subplot(122)
    plt.imshow(out[0,1].squeeze(0).detach().numpy(),cmap='gray')
    plt.title('Segmented Nodules')
    plt.savefig(savePath+'results_slice_'+str(sliceNum)+'.png')
    plt.close()
