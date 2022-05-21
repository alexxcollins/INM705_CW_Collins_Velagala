import os,sys
import random
from pathlib import Path
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import ujson
from matplotlib.patches import Rectangle, Polygon
import warnings
import scipy
import scipy.ndimage


import pycocotools.mask as mask_utils


logger = True

"""
TODO:
idx_to_img: check if any downstream changes needed after idx_to_img() method changes
            method creates the .idx_img_map attribute. Quick test showed it worked
idx_to_img: load negative set  - for eval metrics - DONE
add transforms in load image method when stage = train? (horizontal flip/brightness)
add scores in plot_predictions - ADDED
output class label - class out of index error - FIXED  
see if this show up 
"""

class LVISData(data.Dataset):
    
    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.ds_path = kwargs['ds_path']
        self.labels_f = kwargs['ds_path'] + kwargs['labels_dir'] + \
            '/lvis_v1_{}.json'.format(self.stage)
        self.imgs_dir = kwargs['ds_path'] + kwargs['images_dir'] + \
            '/train2017'  #LVIS train/val/test split not same as coco
        
        self.ann_data = self.get_ann_data(self.labels_f)
        self.classes = self.get_classes_dict(kwargs['classes'])
        self.MAX_IMG_HEIGHT = kwargs['height']
        self.MAX_IMG_WIDTH = kwargs['width']
        self.max_negative = kwargs['max_negative']

        # reindexes from  lvis class # to new value starting from 
        self.class_idx_map = self.map_classes()
        
        # custom class index to image file name, and structure to hold information about 
        # class datasets
        self.idx_img_map, self.class_datasets = self.idx_to_img() 
        #load basic idx to id maps for easy access 
        self.idx_ann_map = self.idx_to_ann() # custom image idx to LVIS annotation id
        self.ann_id_img_map = self.img_id_to_ann_idx()  #image file name to ann_id 
        random.seed(42)
        
        if logger:
            print("stage: ", self.stage)
            print("classes: ", self.classes)
            print("ds_path: ", self.ds_path)
            print("labels_f: ", self.labels_f)
            print("imgs_dir: ", self.imgs_dir)
            
    """
    Re-indexes from LVIS class 3 to new values starting from 1
    cannot use 0 - reserved for background classes 
    """
    def map_classes(self):
        mapped_idx ={}
        temp = {} 
        for i, key in enumerate(self.classes.keys()):
            mapped_idx[self.classes.get(key)] = (i+1) 
            temp[i+1] = key 
        if logger:
            print(f"classes : {temp}")
        
        return mapped_idx 
            
                  
    """
    Returns contents of file 
    """
    def get_ann_data(self, file_name):
        with open(self.labels_f, "r") as f:
            data = ujson.load(f)
        return data 
    
    """
    Enumerates images with classes of interest  
    for stage specified 
    """
    def idx_to_img(self):
        """Return the total dataset. Also return per category datasets
        
        LVIS is a "federated dataset" so we need to keep track of datasets for each category for test purposes.
        
        Return:
            idx_img_map: a dictionary with integer keys and LVIS img id values.
            dictionary with categories as keys, positive, negative and union of positive and negative sets as values
        
        """
        idx_img_map = {} 
        
        #all images in current stage
        # stg_imgs = [f for f in os.listdir(self.imgs_dir) if not f.startswith('.')]
        # stg_imgs = [int(stg_imgs[x].split('.')[0].lstrip('0')) for x in range(0,len(stg_imgs))]
        
        #load positive set 
        pos_imgs = set()
        anns = self.ann_data['annotations']
        classes = set(self.classes.values())
        
        # structure to hold per category dataset
        class_datasets = {}
        for i in range(1, len(self.class_idx_map) + 1): 
            class_datasets[i] = {'union': set(),
                                 'positive': set(),
                                 'negative': set()}
        
        for ann in anns:
            cat_id = ann['category_id']
            img_id = ann['image_id']
            if (cat_id in classes): # and (img_id in stg_imgs):
                pos_imgs.add(img_id) 
                class_datasets[self.class_idx_map[cat_id]]['positive'].add(img_id)
                
        #load negative set and non-exhaustive set
        neg_imgs = set()  # initialise empty negative set
        non_exhaustive = set()
        
        for img in self.ann_data['images']:
            negs = set(img['neg_category_ids'])
            if not negs.isdisjoint(classes):
                # neg_imgs.add(img['id'])
                # add this image to any of our negative sets for our categories
                for cat in negs.intersection(classes):
                    class_datasets[self.class_idx_map[cat]]['negative'].add(img['id'])
                
            n_exhaust = set(img['not_exhaustive_category_ids'])
            if not n_exhaust.isdisjoint(classes):
                non_exhaustive.add(img['id'])
                
        # restrict negative set size to make dataset manageable
        neg_imgs = set(list(neg_imgs)[:self.max_negative])
        
        # we also need to create the union of positive and negative datasets for 
        # individual categories. We remove any non-exhaustively labeled images - we remove
        # all of these. So if img A is non-exhaustive for cats, we still take it away form
        # the dogs set.
        temp = set()
        for cat, d in class_datasets.items():
            d['positive'] = d['positive'] - non_exhaustive
            d['negative'] = d['negative'] - non_exhaustive 
            
            if len(d['negative']) > self.max_negative:
                d['negative'] = set(random.sample(d['negative'],
                                          self.max_negative))
                
            d['union'] = d['positive'].union(d['negative'])
            # add the negative images to our dataset
            neg_imgs = neg_imgs.union(d['negative'])
            # convert sets to lists:
            for k, v in d.items():
                d[k] = list(v)
            
        # create union of positive and negative, remove non-exhaustive 
        imgs = pos_imgs.union(neg_imgs) - non_exhaustive
        
        idx_img_map = dict(zip(range(len(imgs)), imgs))
        idx_img_reverse = dict(zip(imgs, range(len(imgs))))
        
        for cat, d in class_datasets.items():
            # convert all image ids to our new custom id:
            for img in d['negative']:
                img = idx_img_reverse[img]
            for img in d['positive']:
                img = idx_img_reverse[img]
        
        if logger:
            print(f"loaded {len(pos_imgs)} positive set images")
            print(f"loaded {len(neg_imgs)} negative set images")
            print(f"loaded {len(non_exhaustive)} non-exhaustive set images")
            print("Loaded {} images!".format(len(imgs)))
            for key, d in class_datasets.items():
                print('class {} has {} positive and {} negative images'
                      .format(key, len(d['positive']), len(d['negative'])))
    
        return idx_img_map, class_datasets
    
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
        # idx_ann_map = defaultdict(list)
        idx_ann_map = {}
        for i in self.idx_img_map.keys():
            idx_ann_map[i] = []
        
        anns = self.ann_data['annotations']
        imgs = list(self.idx_img_map.values())
        LVIS_to_custom = dict(zip(self.idx_img_map.values(),
                                  self.idx_img_map.keys()))
        classes = list(self.classes.values())
        
        counter = 0 
        
        for ann in anns:
            cat_id = ann['category_id']
            img_id = ann['image_id']
            ann_id = ann['id']
            if(cat_id in classes) and (img_id in imgs):
                # idx = self.get_key_val(self.idx_img_map, img_id)
                idx = LVIS_to_custom[img_id]
                if idx is not None:
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
        
        #needs to process images of diff size,in batch 
        if self.stage == "train" or self.stage == "val":
            #get image metadata 
            img_idx = self.ann_id_img_map[img_id]
            img_height = self.ann_data['images'][img_idx]['height']
            img_width = self.ann_data['images'][img_idx]['width']
            #if logger:
            #    print(f"Loaded image {fname}, old height: {img_height}, old width: {img_width}")
            
            #rescale image 
            ratio = min(self.MAX_IMG_HEIGHT/img_height, self.MAX_IMG_WIDTH/img_width)
            new_height = round(img_height*ratio) 
            new_width = round(img_width*ratio)
            #if logger: 
            #    print(f"Scale factor: {ratio}, new_height: {new_height}, new_width: {new_width}") 
            
        
            bottom_pad = self.MAX_IMG_HEIGHT - new_height 
            right_pad = self.MAX_IMG_WIDTH - new_width
            tfrm = transforms.Compose([transforms.Resize((new_height, new_width)),
                                       transforms.Pad((0,#left
                                                       0,#top 
                                                       (self.MAX_IMG_WIDTH - new_width), #right
                                                      (self.MAX_IMG_HEIGHT - new_height)), #bottom 
                                                      (192,192,192) #color
                                                     ),
                                        transforms.ToTensor()])           
            
        else:
            tfrm = transforms.Compose([transforms.ToTensor()])
        
        img = tfrm(img)
        return img 
    
    """
    Plots image 
    """
    def plot_img(self, idx):
        img_tensor = self.load_img(idx) 
        plt.imshow(img_tensor.permute(1, 2, 0)  )
        return

  
    """
    Given an ann_idx bounding box as a list of coords
    """
    def get_bboxes_by_ann(self, idx, ann_id):
        
        ann = self.ann_data['annotations'][ann_id -1]
        x, y, w, h  = ann['bbox']
        xmax = x + w 
        ymax = y + h 
        
        #if logger:
        #    print(f"bbox: [{x}, {y}, {xmax}, {ymax}] , height: {h}, width: {w}") 
        
        if self.stage == "train" or self.stage == "val":
            #get image id associated with annotation 
            img_id = self.idx_img_map[idx]
            img_idx = self.ann_id_img_map[img_id]
            #get image metadata 
            img_height = self.ann_data['images'][img_idx]['height']
            img_width = self.ann_data['images'][img_idx]['width']
            #scale factor 
            ratio = min(self.MAX_IMG_HEIGHT/img_height, self.MAX_IMG_WIDTH/img_width)
            
            #if logger: 
            #    print(f"Scale factor: {ratio}") 
                
            #new bbox limits
            x = x*ratio
            y = y*ratio
            xmax = x + round(w*ratio)
            ymax = y + round(h*ratio)
            
            
            #if logger:
            #    print(f"new bbox: [{x}, {y}, {xmax}, {ymax}] , height: {round(h*ratio)}, width: {round(w*ratio)}") 
            
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
        
        if self.stage == "train" or self.stage == "val":
            #get scale factor 
            ratio = min(self.MAX_IMG_HEIGHT/h, self.MAX_IMG_WIDTH/w)
            
            #rescale mask for training/validation 
            #https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
            mask = scipy.ndimage.zoom(mask, zoom = ratio, order = 0)
            right_padding = self.MAX_IMG_WIDTH - round(w*ratio) 
            bottom_padding = self.MAX_IMG_HEIGHT - round(h*ratio) 

            mask = np.pad(mask, ((0, bottom_padding), (0, right_padding)), constant_values = 0)
                
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
        
        
        #if ann_ids is not None:
        for ann_id in ann_ids:
            ann_class = annotations[ann_id-1]['category_id']

            if ann_class in classes:
                bbox = self.get_bboxes_by_ann(idx, ann_id)                
                mask = self.get_mask(idx, ann_id)

                bboxes.append(bbox)
                masks.append(mask)


                #reindex classes based on new index values 
                ann_class = self.class_idx_map[ann_class]
                inst_classes.append(ann_class)
                    
        #add handling for negative set 
        if len(inst_classes) == 0: 
            inst_classes.append(0) 
            bboxes.append([0,0,1,1])
            img_size = self.load_img(idx).shape
            masks.append(np.zeros((img_size[0], img_size[1], img_size[2])))
        
        #print(inst_classes)
        #print(bboxes) 
        #print(masks)
            
        
            
        bboxes_t = torch.tensor(bboxes, dtype = torch.float)
        masks_t = torch.tensor(np.array(masks), dtype = torch.uint8)
        classes_t = torch.tensor(inst_classes, dtype = torch.int64)
        
        all_labels = {} 
        
        
        all_labels['boxes'] = bboxes_t
        all_labels['masks'] = masks_t
        all_labels['labels'] = classes_t


        return all_labels

    
    """
    Plots image with bounding boxes and annotations
    """
    def plot_img_with_ann(self, idx, bboxes = False, segs = True):
        
        ax = plt.gca()
        ax.axis('off')
        ann_ids = self.idx_ann_map.get(idx)
        annotations = self.ann_data['annotations']
        
        
        #plots image - handles stage 
        plt.imshow(self.load_img(idx).permute(1,2,0))
        #plt.imshow(self.plot_img(idx))
        
        if bboxes:
            for ann_id in ann_ids:
                b = self.get_bboxes_by_ann(idx, ann_id)
                rect = Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
        if segs:
            for ann_id in ann_ids:  
                m = self.get_mask(idx, ann_id)
                #if logger: print(m.shape)
                img = np.ones( (m.shape[0], m.shape[1], 3) )
                #if logger: print(img.shape)
                color_mask = np.random.random((1, 3)).tolist()[0] #np.array([2.0,166.0,101.0])/255
                for i in range(3):
                    img[:,:,i] = color_mask[i]
                #if logger: print(img.shape) 
                ax.imshow(np.dstack( (img, m*0.5) ))
                #if logger: print(np.dstack( (img, m*0.5) ).shape)
                
        plt.show()
        
    """
    Given index and bounding boxes (list of lists), plots both
    (used for test time - loads image and predicted bounding boxes)
    """
    
    def plot_predictions(self, idx, predictions, show_bboxes = True, show_masks = True, show_scores = True):

            if isinstance(predictions, list):
                predictions = predictions[0]

            img_id = self.idx_img_map[idx]
            fname = str(img_id).zfill(12) + '.jpg'
            path = self.imgs_dir + '/' + fname
            im = PILImage.open(path)
            #print(im.size)
           
            #Plots image 
            plt.imshow(im)

            box_corners = [] 
            
            if show_bboxes:
                bboxes = predictions['boxes'].to('cpu').detach().numpy()
                if len(bboxes) > 0:
                    ax = plt.gca()        
                    ax.axis('off')
                    for id, b in enumerate(bboxes):
                        box_corners.append((b[0],b[1]))
                        rect = Rectangle((b[0],b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
            
            if show_bboxes and show_scores:
                scores = predictions['scores'].to('cpu').detach().numpy()
                for i in range(len(scores)):
                    plt.text(box_corners[i][0],box_corners[i][1],str(round(scores[i], 3)))
                

            if show_masks:
                masks = predictions['masks'].to('cpu').detach().numpy()
                if len(masks) > 0:
                    for m in masks:
                        m = m[0, :, :]
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        color_mask = np.random.random((1,3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack((img, m*0.5)))
            

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
         return idx,X,y
        
    
   

                

                     
    

                    
                    
                    
                    
                    
                
                
                
        
        

        

        
        
    
    
    
    
        
        
        
        

        
    