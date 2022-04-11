import os,sys
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Rectangle, Polygon

import pycocotools.mask as mask_utils


logger = True

"""
TODO:
idx_to_img: load negative set  - for eval metrics??
add transforms in load image method when stage = train? 

"""

class LVISData(data.Dataset):
    
    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.ds_path = kwargs['ds_path']
        self.labels_f = kwargs['ds_path'] + kwargs['labels_dir'] +  '/' +"lvis_v1_{}.json".format(self.stage)
        self.imgs_dir = kwargs['ds_path'] + kwargs['images_dir'] + '/' + self.stage + '2017'
        
        self.ann_data = self.get_ann_data(self.labels_f)
        self.classes = self.get_classes_dict(kwargs['classes'])
        
        #load basic idx to id maps for easy access 
        self.idx_img_map = self.idx_to_img()
        self.idx_ann_map = self.idx_to_ann() 
        self.ann_id_img_map = self.img_id_to_ann_idx() 


        
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
    Enumerates images with classes of interest  
    for stage specified 
    """
    def idx_to_img(self):
        idx_img_map = {} 
        
        #all images in current stage
        
        stg_imgs = [f for f in os.listdir(self.imgs_dir) if not f.startswith('.')]
        stg_imgs = [int(stg_imgs[x].split('.')[0].lstrip('0')) for x in range(0,len(stg_imgs))]
        
        
        #load positive set 
        imgs = [] 
        anns = self.ann_data['annotations']
        classes = list(self.classes.values())        
        for ann in anns:
            cat_id = ann['category_id']
            img_id = ann['image_id']
            if (cat_id in classes) and (img_id in stg_imgs):
                imgs.append(img_id)     
                
        #remove duplicates 
        imgs = list(set(imgs))
        
        for idx, image_id in enumerate(imgs):
            idx_img_map[idx] = image_id 
            
        if logger:
            print("Loaded {} images!".format(len(imgs)))
    
        return idx_img_map
    
    """
    Helper function: returns idx given image id 
    """
    def get_key_val(self, d, v):
        for key, val in d.items():
            if v == val:
                return key 
        return None 
        
    
    """
    Returns dict of all annotations ids associated with each index 
    """
    def idx_to_ann(self):
        idx_ann_map = defaultdict(list)
        
    
        anns = self.ann_data['annotations']
        imgs = list(self.idx_img_map.values())
        classes = list(self.classes.values())
        
        counter = 0 
        
        for ann in anns:
            cat_id = ann['category_id']
            img_id = ann['image_id']
            ann_id = ann['id']
            if(cat_id in classes) and (img_id in imgs):
                idx = self.get_key_val(self.idx_img_map, img_id)
                if idx:
                    idx_ann_map[idx].append(ann_id)
                    counter += 1 
                        
        if logger:
            print("{} annotations found!".format(counter))
            
        return dict(sorted(idx_ann_map.items()))
    
    """
    Constructs dictionary of:
    keys: image ids 
    value: index of key in annotation[images]
    """
    def img_id_to_ann_idx(self):
        all_imgs = self.ann_data['images']
        class_imgs = list(self.idx_img_map.values()) 
        
        img_id_ann_idx_map = {} 
        
        for idx, img in enumerate(all_imgs):
            if img['id'] in class_imgs:
                img_id_ann_idx_map[img['id']] = idx
                
        return img_id_ann_idx_map
            
    """
    Returns dictionary of classes and ids from annotations  
    for specified classes 
    """
    def get_classes_dict(self, classes):
        categories = self.ann_data['categories']
        
        classes_dict = {}
        for cat in categories:
            if cat['name'] in classes:
                classes_dict[cat['name']] = cat['id']
        return classes_dict
    
       
    """
    Returns image as a tensor
    """
    def load_img(self, idx):
        
        try: 
            img_id = self.idx_img_map[idx]
        except KeyError as e:
            raise e
        
        fname = str(img_id).zfill(12) + '.jpg'
        path = self.imgs_dir + '/' + fname
        img = PILImage.open(path).convert("RGB")
        
        tfrm = transforms.Compose([transforms.ToTensor()])
        img = tfrm(img)
        return img 
    
    
  
    """
    Given an ann_idx bounding box as a list of coords
    """
    def get_bboxes_by_ann(self, ann_id):
        
        ann = self.ann_data['annotations'][ann_id -1]            
        x, y, w, h  = ann['bbox']
        xmax = x + w 
        ymax = y + h 
 
        return [x,y, xmax, ymax]

    """
    Returns mask given class image idx and ann_id 
    uses pycoco methods for decompressing RLE
    """
    def get_mask(self, idx, ann_id):
        
        img_id = self.idx_img_map[idx]
        ann_img_idx = self.ann_id_img_map.get(img_id)
        
        img_data = self.ann_data['images'][ann_img_idx]
        h, w = img_data["height"], img_data["width"]
        seg = self.ann_data['annotations'][ann_id-1]['segmentation']
        
        rle = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rle)
            
        mask =  mask_utils.decode(rle)
        
        return mask
    
    
    """
    Given an index, 
    returns labels, bboxes, masks as tensors 
    """
    def get_label(self, idx, classes = None):
         
        ann_ids = self.idx_ann_map.get(idx)
            
        annotations = self.ann_data['annotations']
        
        #if classes are not specified
        #returns all classes dataset is initialized with
        if not(classes): 
            classes = list(self.classes.values())
        
        inst_classes = []
        bboxes = []
        masks = [] 
        
        for ann_id in ann_ids:
            ann_class = annotations[ann_id-1]['category_id']
            
            if ann_class in classes:
                bbox = self.get_bboxes_by_ann(ann_id)                
                mask = self.get_mask(idx, ann_id)
                
                bboxes.append(bbox)
                masks.append(mask)
                
                ##remove this later 
                #temp = {959: 1, 982: 2}
                #ann_class = temp[ann_class]
                
                
                inst_classes.append(ann_class)
            
        bboxes_t = torch.tensor(bboxes, dtype = torch.float)
        masks_t = torch.tensor(masks, dtype = torch.uint8)
        classes_t = torch.tensor(inst_classes, dtype = torch.int64)
        
        all_labels = {} 
        
        #all_labels['bboxes'] = bboxes_t
        #all_labels['masks'] = masks_t
        #all_labels['classes'] = classes_t
        
        all_labels['boxes'] = bboxes_t
        all_labels['masks'] = masks_t
        all_labels['labels'] = classes_t


        return all_labels


    """
    Plots image 
    """
    def plot_img(self, idx):
        
        try: 
            img_id = self.idx_img_map[idx]
        except KeyError as e:
            raise e
        
        fname = str(img_id).zfill(12) + '.jpg'
        path = self.imgs_dir + '/' + fname
        return PILImage.open(path)
    

    
    """
    Plots image with bounding boxes and annotations
    """
    def plot_img_with_ann(self, idx, bboxes = False, segs = True):
        
        ax = plt.gca()
        ax.axis('off')
        ann_ids = self.idx_ann_map.get(idx)
        annotations = self.ann_data['annotations']
        
        
        #plots image
        plt.imshow(self.plot_img(idx))
        
        if bboxes:
            for ann_id in ann_ids:
                b = self.get_bboxes_by_ann(ann_id)
                rect = Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
        if segs:
            for ann_id in ann_ids:            
                m = self.get_mask(idx, ann_id)
                img = np.ones( (m.shape[0], m.shape[1], 3) )
                color_mask = np.random.random((1, 3)).tolist()[0] #np.array([2.0,166.0,101.0])/255
                for i in range(3):
                    img[:,:,i] = color_mask[i]
                ax.imshow(np.dstack( (img, m*0.5) ))
                
        plt.show()
        
    """
    Given index and bounding boxes (list of lists), plots both
    (used for test time - loads image and predicted bounding boxes)
    """
    
    def plot_img_bboxes(self,idx, bboxes):
        
        img_id = self.idx_img_map[idx]
        fname = str(img_id).zfill(12) + '.jpg'
        path = self.imgs_dir + '/' + fname
        im = PILImage.open(path)
        
        
        if len(bboxes) > 0:
            plt.imshow(im)
            ax = plt.gca()        
            ax.axis('off')
            for id, b in enumerate(bboxes):
                #b = b[0] #get tuple
                rect = Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            
        plt.show()
        return 


    """
    Number of images class instantiated with 
    """
    def __len__(self):
         return len(self.idx_img_map)

    """
    magic method for iterating class items
    """
    def __getitem__(self, idx):
         X = self.load_img(idx)
         y = self.get_label(idx) 
         return X,y
        
    
    
    ##############DONT NEED THIS ANYMORE###########################33
        
    """
    Given image index and class ids, 
    returns dictionary of classes (keys) and bounding boxes (list of tuples)
    {'a': [[x1,y1,x2,y2], [x1,y1,x2,y2]], 'b' : [[x1,...]]}
    """

    def get_bounding_boxes(self, idx, classes):
        
        ann_ids = self.idx_ann_map.get(idx)
        annotations = self.ann_data['annotations']
        classes = list(self.classes.values())
        
        class_bboxes_dict = defaultdict(list)
    
        
        for ann_id in ann_ids:
            ann = annotations[ann_id -1]            
            ann_class = ann['category_id']
            
            if ann_class in classes:
                x, y, w, h  = ann['bbox']
                xmax = x + w 
                ymax = y + h 
                class_bboxes_dict[ann_class].append([(x,y, xmax, ymax)])
 
        return class_bboxes_dict


  
    
        

                

                     
    

                    
                    
                    
                    
                    
                
                
                
        
        

        

        
        
    
    
    
    
        
        
        
        

        
    