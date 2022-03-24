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

"""
TODO:
1. get_labels: update to return masks 


"""

class LVISData(data.Dataset):
    
    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.ds_path = kwargs['ds_path']
        self.labels_f = kwargs['ds_path'] + kwargs['labels_dir'] +  '/' +"lvis_v1_{}.json".format(self.stage)
        self.imgs_dir = kwargs['ds_path'] + kwargs['images_dir'] + '/' + self.stage + '2017'
        self.ann_data = self.get_ann_data(self.labels_f)
        self.classes = self.get_classes_dict(kwargs['classes'])

        
        if logger:
            print("stage: ", self.stage)
            print("classes: ", self.classes)
            print("ds_path: ", self.ds_path)
            print("labels_f: ", self.labels_f)
            print("imgs_dir: ", self.imgs_dir)
            
    
    """
    Returns contents of file 
    """
    def get_ann_data(self, file_name):
        f = open (self.labels_f, "r")
        data = json.loads(f.read())
        f.close() 
        
        return data 
        
            
    """
    Returns dictionary of classes and ids from annotations  
    for specified classes 
    """
    def get_classes_dict(self, classes):
        #f = open (self.labels_f, "r")
        #data = json.loads(f.read())
        categories = self.ann_data['categories']
        
        classes_dict = {}
        for cat in categories:
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
    
    """
    Given images index, 
    returns annotation ids of image 
    (filtered to classes, if specified)
    """
    def get_ann_ids(self, idx, classes = None):
        #f = open (self.labels_f, "r")
        #data = json.loads(f.read())
        
        annotations = self.ann_data['annotations']
        
        class_ids = [] 
        ann_ids = [] 
        
        for ann in annotations:
            if ann['image_id'] == idx:
                ann_ids.append(ann['id'])
                class_ids.append(ann['category_id'])
                
        if logger:
            print(len(ann_ids), "ann_ids: ", ann_ids)
            print(len(ann_ids), "class_ids: ", class_ids)
                
        if classes:
            popped = 0 
            for i in range(0,len(class_ids)):
                if class_ids[i] not in classes:
                    ann_ids.pop(i - popped)
                    popped += 1
        
        return ann_ids 
    
    """
    Given an index, 
    returns labels, bboxes, masks as tensors 
    """
    def get_labels(self, idx, classes = None):
         
        ann_ids = self.get_ann_ids(idx, classes)
        
        class_bboxes_dict = {}
        
        #f = open (self.labels_f, "r")
        #data = json.loads(f.read())['annotations']
        annotations = self.ann_data['annotations']
        
        bboxes = []
        classes = []
        masks = [] 
        
        for ann_id in ann_ids:
            ann_class = annotations[ann_id-1]['category_id']
            
            if classes:
                if ann_class in classes:
                    x, y, w, h = annotations[ann_id-1]['bbox'] #annotation ids start at 1 
                    xmax = x + w 
                    ymax = y + h 
                    bboxes.append([x,y, xmax, ymax])
                    masks.append(annotations[ann_id-1]['segmentation'])
                    classes.append(ann_class)
            else:
                x, y, w, h = annotations[ann_id-1]['bbox'] #annotation ids start at 1 
                xmax = x + w 
                ymax = y + h 
                bboxes.append([x,y, xmax, ymax])
                masks.append(annotations[ann_id-1]['segmentation'])
                classes.append(ann_class)
        
        bboxes_t = torch.tensor(bboxes, dtype = torch.float)
        #masks_t = torch.tensor(masks, dtype = torch.float)
        classes_t = torch.tensor(classes, dtype = torch.int)
        
        all_labels = {} 
        
        all_labels['bboxes'] = bboxes_t
        #all_labels['masks'] = masks_t
        all_labels['classes'] = classes_t

        return all_labels

                     
    
    """
    Given image index and class ids, 
    returns dictionary of classes (keys) and bounding boxes (list of tuples)
    {'a': [[x1,y1,x2,y2], [x1,y1,x2,y2]], 'b' : [[x1,...]]}
    """

    def get_bounding_boxes(self, idx, classes):
        #f = open (self.labels_f, "r")
        #data = json.loads(f.read())
        
        annotations = self.ann_data['annotations']
        
        class_bboxes_dict = {}
        
        for ann in annotations['annotations']:
            
            ann_img_id = ann['image_id']
            ann_class = ann['category_id']
            
            if (ann_img_id == idx)  and  (ann_class in classes):
                x, y, w, h  = ann['bbox']
                xmax = x + w 
                ymax = y + h 
                
                #new entry 
                if ann['category_id'] not in class_bboxes_dict:
                    class_bboxes_dict[ann_class] = [(x,y, xmax, ymax)]
                #if key already exists
                elif isinstance(class_bboxes_dict[ann_class], list):
                    class_bboxes_dict[ann_class].append((x,y, xmax, ymax))
 
        return class_bboxes_dict
                
                    
                    
                    
                    
                    
                
                
                
        
        

        

        
        
    
    
    
    
        
        
        
        

        
    