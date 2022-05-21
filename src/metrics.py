import torch 
from collections import Counter

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

def store_preds(idx, y, y_pred, pred_boxes = [], gt = []):
    """call this in test loop. It will create:
      - a list of predictions
      - a list of ground truths
      which we use for mAP
    
    assumes test batch size is 1
    """
    scores = y_pred[0]['scores']
    labels = y_pred[0]['labels']
    boxes = y_pred[0]['boxes'].detach().cpu()   # need to make sure this tensor is correct
    idx = idx[0] # returns int
 
    for i in range(len(labels)):
        list_ = [idx,
                 labels[i].item(),
                 scores[i].item(),
                 boxes[i]
                ]
        pred_boxes.append(list_)
        
    labels = y[0]['labels']
    boxes = y[0]['boxes']
    
    for i in range(len(labels)):
        list_ = [idx,
                 labels[i].item(),
                 boxes[i]
                ]
        gt.append(list_)
    
    return pred_boxes, gt


def calculate_ap(pred_box, gt_boxes, num_classes, iou_threshold=0.5):
    """
    Calculating AP for a given class for a given IOU threshold 
    
    input:
    pred_box: list for each prediction consisting of:
        [img_idx, label, score, box: tensor]
    gts: list of ground truths for all images in test set consisting of:
        [img_idx, label, box: tensor]
    """
    
    epsilon = 1e-6
    AP = []
    print(len(pred_box))
    
    for c in range(1, num_classes + 1):
        detections = []
        gts = []
        
        for detection in pred_box:
            if detection[1] == c:
                detections.append(detection)
        print(f'class {c}: len detections: {len(detections)}')
                
        for g in gt_boxes:
            if g[1] == c:
                gts.append(g)
                
        # count how many ground truth bboxes we have for each image
        amount_bboxes = Counter([gt[0] for gt in gts])
        
        # create variable length tensor to help keep track of when 
        # we have prediction matched to a gt so we don't repeat TPs
        for k, v in amount_bboxes.items():
            amount_bboxes[k] = torch.zeros(v)
            
        # sort detections in reverse conficence order
        detections.sort(key = lambda x: x[2], reverse=True)
        
        # structures to keep track of TP vs FP as we go down predictions
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        tot_true_bboxes = len(gts) # denominator for recall
        
        for det_idx, detection in enumerate(detections):
            # create list of gt bboxes for image in detection
            gt_img = [bbox for bbox in gts if bbox[0] == detection[0]]
        
            best_iou = 0
            num_gts = len(gt_img)
            
            for gt_idx, gt in enumerate(gt_img):
                iou = get_iou(detection[3], gt[2]).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > iou_threshold: # then we have TP
                # check we don't alreay have TP marked against this GT:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[det_idx] = 1
                    # note that GT has TP held against it:
                    amount_bboxes[detection[0]][best_gt_idx] = 1 
                else: # FP
                    FP[det_idx] = 1
            else: # FP
                FP[det_idx] = 1
        
        print(f'for category {c}:\n-------')
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        print('TP and FP:')
        print(TP_cumsum)
        print(FP_cumsum)
        recalls = TP_cumsum / (tot_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        print(f'total gts {tot_true_bboxes}')
        print('precisions and recalls:')
        print(precisions)
        print(recalls)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        AP.append(torch.trapz(precisions, recalls))
        print(f'AP is {AP[-1].item()}')
        print('-------\n')
        
    per_cls_AP = dict(zip(range(1, num_classes + 1), AP))
    
    return sum(AP) / len(AP), per_cls_AP
    
    #for each training example find number of bboxes 
    
    
    

def calculate_map(preds, gt, iou_threshold, num_classes):
    
    #for each class 
        #go through predictions and see if prediction is of class we're interested in 
        
        #go through each gt and see if prediction is of class we're interested in 
        
        #create a dictionary for each image, # of gt bboxes 
    pass
        