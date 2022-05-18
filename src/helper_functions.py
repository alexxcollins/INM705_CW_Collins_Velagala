import torch 
import os 
import sys 
from pathlib import Path 

CHECKPOINT_DIRECTORY = "model_checkpoints"



"""
TODO:
-add load latest checkpoint function 

"""

"""
custom class to create batches 
"""

class CollateCustom:
    
    def __call__(self, batch):
        
        idx = [item[0] for item in batch]
        
        X = [item[1].unsqueeze(0) for item in batch]
        X = torch.cat(X, dim = 0)
        
        y = [item[2] for item in batch]
        
        return idx, X, y
    

"""
saves model weights + optimizer params + epoch 
"""
def save_checkpoint(checkpoint, fname):
    path = Path.cwd().parent.joinpath(CHECKPOINT_DIRECTORY).joinpath(fname)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint {fname}!") 
    
"""
reload model from checkpoint
"""
def load_checkpoint(fname, model, optimizer):
    path = Path.cwd().parent.joinpath(CHECKPOINT_DIRECTORY).joinpath(fname)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded checkpoint {fname}!")

    
"""
For inference
saves in the main directory, may need to change later? 
""" 
def save_model(model, fname):
    torch.save(model, fname) 
    print(f"Saved model {fname}!")
    
def load_model(model, fname):
    model.load_state_dict(torch.load(fname)) 
    print(f"Loaded from model {fname}!")

    
"""
Filters model output predictions and ground truths to specified class
Can input multiple images 

"""
    
def filter_to_label(predictions, ground_truth, class_label): 
    
    gts = [] 
    
    #key = image idx - for each image 
    for i, key in enumerate(ground_truth.keys()):
    
        labels = ground_truth.get(key)['labels'].to('cpu').detach()
        boxes = ground_truth.get(key)['boxes'].to('cpu').detach()
        
        bboxes = []
        #get bbboxes of interest
        for i, label in enumerate(labels):
            if label == class_label:
                bboxes.append(boxes[i]) 
                
        gts.append([key, bboxes])
    
    #print("Ground Truth:")
    #print(gts) 
    
    
    
    preds = [] 
    
    for i, key in enumerate(predictions.keys()):
        
        labels = predictions.get(key)['labels'].to('cpu').detach()
        boxes = predictions.get(key)['boxes'].to('cpu').detach()
        scores = predictions.get(key)['scores'].to('cpu').detach()
        
        bboxes =[] 
        conf_scores = [] 
        
        for i, label in enumerate(labels):
            if label == class_label:
                bboxes.append(boxes[i])
                conf_scores.append(scores[i]) 
                
        preds.append([key, scores, boxes])
        
    #print("Predictions:")
    #print(preds)
        
    return gts, preds  
        
    
    