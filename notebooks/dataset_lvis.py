import os,sys
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import json

logger = True

class LVISData(data.Dataset):
    
    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.ds_path = kwargs['ds_path']
        self.labels_f = kwargs['ds_path'] + kwargs['labels_dir'] +  '/' +"lvis_v1_{}.json".format(self.stage)
        self.imgs_dir = kwargs['ds_path'] + kwargs['images_dir'] + '/' + self.stage + '2017'
        self.classes = self.get_classes_dict(kwargs['classes'])

        
        if logger:
            print("stage: ", self.stage)
            print("classes: ", self.classes)
            print("ds_path: ", self.ds_path)
            print("labels_f: ", self.labels_f)
            print("imgs_dir: ", self.imgs_dir)
            
    """
    Returns dictionary of classes and ids from annotations  
    for specified classes 
    """
    def get_classes_dict(self, classes):
        f = open (self.labels_f, "r")
        data = json.loads(f.read())
        classes_dict = {}
        for cat in data['categories']:
            if cat['name'] in classes:
                classes_dict[cat['name']] = cat['id']
        f.close()
        return classes_dict
    
    """
    Returns image as a tensor
    """
    def load_img(self, idx):
        fname = str(idx).zfill(12) + '.jpg'
        path = self.imgs_dir + '/' + fname
        img = np.array(PILImage.open(path))
        tfrm = transforms.Compose([ transforms.ToPILImage(),  transforms.ToTensor()])
        img = tfrm(img)
        return img 
    
    """
    Plots image 
    """
    def plot_img(self, idx):
        fname = str(idx).zfill(12) + '.jpg'
        path = self.imgs_dir + '/' + fname
        return PILImage.open(path)
        

        

        
        
    
    
    
    
        
        
        
        

        
    