import torch 
import os 

CHECKPOINT_DIRECTORY = "model_checkpoints"



"""
saves model weights + optimizer params + epoch 
"""
def save_checkpoint(checkpoint, fname):
    path = CHECKPOINT_DIRECTORY + "/" + fname 
    torch.save(checkpoint, path)
    print(f"Saved checkpoint {fname}!") 
    
"""
reload model from checkpoint
"""
def load_checkpoint(fname, model, optimizer):
    path =  CHECKPOINT_DIRECTORY + "/" + fname 
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded checkpoint {fname}!")

"""
For inference
""" 
def save_model(model, fname):
    torch.save(model, fname) 
    print(f"Saved model {fname}!")
    
def load_model(model, fname):
    model.load_state_dict(torch.load(fname)) 
    print(f"Loaded from model {fname}!")
    