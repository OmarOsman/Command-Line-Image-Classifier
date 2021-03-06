import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import copy
import torch
import time
import json
import pdb
from tqdm import tqdm
from tqdm import tnrange


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils ,datasets,models
from collections import OrderedDict
from PIL import Image


def process_image(image_path,device):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a torch tensor
    '''

    mean,std = [0.485, 0.456, 0.406] ,[0.229, 0.224, 0.225]
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image_path)
    
    # Suggestons from the reviewer , hanlding the aspect ratio  
    aspect_ratio = im.size[0]/im.size[1]
    if aspect_ratio > 0:
        im.thumbnail((999999, 256))
    else:
        im.thumbnail((256, 999999))
    
    # Suggestons from the reviewer  ,Crop the center of the image
    width, height = im.size   # Get dimensions
    left   = (width -224)/ 2
    bottom = (height - 224)/2
    
    right = left + 224
    top   = bottom + 224
    im = im.crop((left, bottom, right,top))
    
    # Convert to 0 1 range
    np_image = np.array(im).astype(np.float32)
    np_image/=225.0
    
    # Normalization
    np_image = (np_image - mean) / std
    
    # transpose 
    np_image = np_image.transpose(2,0,1)
    tensor = torch.tensor(np_image, device=device).float()
    return tensor


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
