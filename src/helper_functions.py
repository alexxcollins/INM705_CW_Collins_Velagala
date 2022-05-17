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
    