import torch 


"""
Return IOU between all predictions and one GT 
Preds: [x1, y1, x2, y2] (tensor) 
GT: [x1, y1, x2, y2] (tensot) 
"""
def get_iou(preds, gt):
    
    #get corners of predictions 
    pred_x1 = preds[..., 0:1] 
    pred_y1 = preds[..., 1:2]
    pred_x2 = preds[..., 2:3]
    pred_y2 = preds[..., 3:4]

    #get corner of GT 
    gt_x1 = gt[..., 0:1]
    gt_y1 = gt[..., 1:2]
    gt_x2 = gt[..., 2:3]
    gt_y2 = gt[..., 3:4]

    #get corners of intersection 
    x1 = torch.max(pred_x1, gt_x1)
    y1 = torch.max(pred_y1, gt_y1)
    x2 = torch.min(pred_x2, gt_x2)
    y2 = torch.min(pred_y2, gt_y2)

    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0) 
    pred_box_areas = abs((pred_x2-pred_x1) * (pred_y2-pred_y1))
    gt_box_area = abs((gt_x2-gt_x1) * (gt_y2-gt_y1))
    union = pred_box_areas + gt_box_area - intersection + 1e-6 #avoid divide by 0 

    return (intersection/union)

"""
Calculating AP per class for a given IOU threshold 
input:
preds: [img_idx, conf_score, [[x1,y1, x2,y2], [x1,y1,x2,y2], ....]]
gt: [img_idx, [[x1,y1, x2,y2], [x1,y1,x2,y2], ....]
"""
def calculate_ap(preds, gts, iou_threshold):
    
    #for each training example find number of bboxes 
    
    
    

def calculate_map(preds, gt, iou_threshold, num_classes):
    
    #for each class 
        #go through predictions and see if prediction is of class we're interested in 
        
        #go through each gt and see if prediction is of class we're interested in 
        
        #create a dictionary for each image, # of gt bboxes 
        