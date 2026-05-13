import torch
import numpy as np
import SimpleITK as sitk
from scipy.optimize import linear_sum_assignment


def mean_dice(pred, target, num_classes=3):
    dice_scores = []
    for class_id in range(num_classes):
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        intersection = np.sum(pred_mask * target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)
        dice_score = (2.0 * intersection) / (union + 1e-7)  # Add a small epsilon to avoid division by zero
        dice_scores.append(dice_score)
    mean_dice = np.mean(dice_scores)
    return mean_dice

def mean_iou(pred, target, num_classes=3):
    iou_scores = []
    for class_id in range(num_classes):
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        intersection = np.sum(pred_mask * target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask) - intersection
        iou_score = intersection / (union + 1e-7)  # Add a small epsilon to avoid division by zero
        iou_scores.append(iou_score)
    mean_iou = np.mean(iou_scores)
    return mean_iou

# def compute_mean_dice(labels_true, labels_pred, num_classes):
#     mean_dice = 0.0
#
#     for class_id in range(num_classes):
#         if labels_true == 2:
#             print("good")
#         intersection = np.sum((labels_true == class_id) & (labels_pred == class_id))
#         dice = (2.0 * intersection) / (np.sum(labels_true == class_id) + np.sum(labels_pred == class_id))
#         mean_dice += dice
#
#     mean_dice /= num_classes
#     return mean_dice
#
# def compute_mean_iou(labels_true, labels_pred, num_classes):
#     mean_iou = 0.0
#
#     for class_id in range(num_classes):
#         intersection = np.sum((labels_true == class_id) & (labels_pred == class_id))
#         union = np.sum((labels_true == class_id) | (labels_pred == class_id))
#         iou = intersection / union
#         mean_iou += iou
#
#     mean_iou /= num_classes
#     return mean_iou

def calculate_miou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''

    inputTmp = torch.zeros([input.shape[0],classNum, input.shape[1],input.shape[2]]).cuda()#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    # target = target.unsqueeze(1)#同上
    input = input.to(torch.int64)
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    target = target.to(torch.int64)
    target = target.unsqueeze(1)  # 同上
    target = target.to('cuda')
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            # if union == 1e-6:
            #     continue
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    return np.mean(batchMious)




def calculate_pq(input, target, classNum=2, iou_threshold=0.5):
    """
    Calculate Panoptic Quality (PQ)
    :param input: [b,h,w] predicted labels (0: background, 1: nucleus)
    :param target: [b,h,w] ground truth labels (0: background, 1: nucleus)
    :param classNum: number of classes (including background)
    :param iou_threshold: IoU threshold for matching
    :return: PQ, SQ, DQ
    """
    batch_pq = []
    batch_sq = []
    batch_dq = []

    for b in range(input.shape[0]):
        pred = input[b].cpu().numpy()
        gt = target[b].cpu().numpy()

        # Get connected components for instances (excluding background)
        pred_instances = _get_instances(pred, classNum)
        gt_instances = _get_instances(gt, classNum)

        if len(gt_instances) == 0:
            # No ground truth nuclei
            if len(pred_instances) == 0:
                batch_pq.append(1.0)
                batch_sq.append(1.0)
                batch_dq.append(1.0)
            else:
                batch_pq.append(0.0)
                batch_sq.append(0.0)
                batch_dq.append(0.0)
            continue

        # Calculate IoU matrix between predictions and ground truth
        iou_matrix = np.zeros((len(pred_instances), len(gt_instances)))
        for i, pred_mask in enumerate(pred_instances):
            for j, gt_mask in enumerate(gt_instances):
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union > 0:
                    iou_matrix[i, j] = intersection / union

        # Match using Hungarian algorithm
        matched_pairs = []
        matched_preds = set()
        matched_gts = set()

        # Find matches with IoU >= threshold
        for i in range(len(pred_instances)):
            for j in range(len(gt_instances)):
                if iou_matrix[i, j] >= iou_threshold:
                    matched_pairs.append((i, j, iou_matrix[i, j]))

        # Sort by IoU descending and assign unique matches
        matched_pairs.sort(key=lambda x: x[2], reverse=True)
        for i, j, iou in matched_pairs:
            if i not in matched_preds and j not in matched_gts:
                matched_preds.add(i)
                matched_gts.add(j)

        # Calculate DQ, SQ, PQ
        tp = len(matched_preds)
        fp = len(pred_instances) - tp
        fn = len(gt_instances) - tp

        dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0

        # Calculate SQ (average IoU of matched pairs)
        if tp > 0:
            sq_sum = 0
            for i, j, _ in matched_pairs:
                if i in matched_preds and j in matched_gts:
                    sq_sum += iou_matrix[i, j]
            sq = sq_sum / tp
        else:
            sq = 0

        pq = dq * sq

        batch_pq.append(pq)
        batch_sq.append(sq)
        batch_dq.append(dq)

    return np.mean(batch_pq), np.mean(batch_sq), np.mean(batch_dq)


def calculate_aji(input, target, classNum=2):
    """
    Calculate Aggregated Jaccard Index (AJI)
    :param input: [b,h,w] predicted labels (0: background, 1: nucleus)
    :param target: [b,h,w] ground truth labels (0: background, 1: nucleus)
    :param classNum: number of classes (including background)
    :return: AJI score
    """
    batch_aji = []

    for b in range(input.shape[0]):
        pred = input[b].cpu().numpy()
        gt = target[b].cpu().numpy()

        # Get connected components for instances
        pred_instances = _get_instances(pred, classNum)
        gt_instances = _get_instances(gt, classNum)

        if len(gt_instances) == 0:
            if len(pred_instances) == 0:
                batch_aji.append(1.0)
            else:
                batch_aji.append(0.0)
            continue

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_instances), len(gt_instances)))
        for i, pred_mask in enumerate(pred_instances):
            for j, gt_mask in enumerate(gt_instances):
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union > 0:
                    iou_matrix[i, j] = intersection / union

        # Match using Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Negative for maximization

        # Calculate AJI
        sum_inter = 0
        sum_union = 0
        matched_preds = set()
        matched_gts = set()

        # Sum for matched pairs
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] > 0:
                pred_mask = pred_instances[i]
                gt_mask = gt_instances[j]
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                sum_inter += intersection
                sum_union += union
                matched_preds.add(i)
                matched_gts.add(j)

        # Add unmatched predictions (false positives)
        for i in range(len(pred_instances)):
            if i not in matched_preds:
                sum_union += pred_instances[i].sum()

        # Add unmatched ground truths (false negatives)
        for j in range(len(gt_instances)):
            if j not in matched_gts:
                sum_union += gt_instances[j].sum()

        aji = sum_inter / sum_union if sum_union > 0 else 0
        batch_aji.append(aji)

    return np.mean(batch_aji)


def _get_instances(mask, classNum):
    """
    Extract individual instances from semantic segmentation mask
    :param mask: [h,w] numpy array (0: background, 1: nucleus)
    :param classNum: number of classes
    :return: list of binary masks for each instance
    """
    from skimage import measure

    instances = []
    # Get connected components for class 1 (nucleus)
    if classNum > 1:
        binary_mask = (mask == 1).astype(np.uint8)
        labeled_mask = measure.label(binary_mask, connectivity=2)

        for instance_id in range(1, labeled_mask.max() + 1):
            instance_mask = (labeled_mask == instance_id)
            instances.append(instance_mask)

    return instances


# Usage example:
# predicted: [b,h,w] tensor
# masks_cuda: [b,h,w] tensor
#
# # Calculate mDice (your existing function)
# train_mdice += Miou.calculate_mdice(predicted, masks_cuda, 2).item()
#
# # Calculate PQ, AJI, DQ, SQ
# pq, sq, dq = calculate_pq(predicted, masks_cuda, classNum=2, iou_threshold=0.5)
# aji = calculate_aji(predicted, masks_cuda, classNum=2)
#
# print(f"PQ: {pq:.4f}, DQ: {dq:.4f}, SQ: {sq:.4f}, AJI: {aji:.4f}")


def calculate_mdice(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]]).cuda()#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = 2 * torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) + 1e-6
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    return np.mean(batchMious)


def Pa(input, target):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    tmp = input == target
    x=torch.sum(tmp).float()
    y=input.nelement()
    # print('x',x,y)
    return (x / y)
def pre(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP+1e-6)/(TP+FP+1e-6)
    return pre
def recall(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    recall=(TP+1e-6)/(TP+FN+ 1e-6)
    return recall
def F1score(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP+1e-6) / (TP + FP + 1e-6)
    recall=(TP+1e-6)/(TP+FN+ 1e-6)
    F1score=(2*(pre)*(recall))/(pre+recall+1e-6)
    return F1score

def jaccard(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    ja = TP/((TP+FN+FP)+1e-5)
    return ja

def accuracy(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    AC = (TP+TN)/((TP+FP+TN+FN+1e-5))
    return AC

def dice(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    DI = 2*TP/((2*TP+FN+FP+1e-5))
    return DI

def recall(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    SE = TP/(TP+FN+1e-5)
    return SE

def SP(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    SP = TN/((TN+FP)+1e-5)
    return SP

def precision(input, target):
    input=input.data.cpu().numpy()
    target=target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    precision = TP/(TP+FP+1e-5)
    return precision






def calculate_fwiou(input,target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]]).cuda()#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]]).cuda()#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)#同上
    batchFwious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        fwious = []
        for j in range(classNum):#遍历类别，包括背景
            TP_FN = torch.sum(targetOht[i][j])
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            if union == 1e-6:
                continue
            iou = intersection / union
            fwiou = (TP_FN/(input.shape[2]*input.shape[3])) * iou
            fwious.append(fwiou.item())
        fwiou = np.mean(fwious)#计算该图像的miou
        # print(miou)
        batchFwious.append(fwiou)
    return np.mean(batchFwious)

def calculate_dice(pred, target):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection) / (union + 1e-8)
    return dice


def calculate_jaccard(pred, target):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    jaccard = intersection / (union + 1e-8)
    return jaccard


def calculate_asd(pred, target, spacing):
    pred_sitk = sitk.GetImageFromArray(pred)
    target_sitk = sitk.GetImageFromArray(target)
    pred_sitk.SetSpacing(spacing)
    target_sitk.SetSpacing(spacing)
    hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_image_filter.Execute(pred_sitk > 0, target_sitk > 0)
    asd = hausdorff_distance_image_filter.GetAverageSurfaceDistance()
    return asd


def calculate_hd(pred, target, spacing):
    pred_sitk = sitk.GetImageFromArray(pred)
    target_sitk = sitk.GetImageFromArray(target)
    pred_sitk.SetSpacing(spacing)
    target_sitk.SetSpacing(spacing)
    hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_image_filter.Execute(pred_sitk > 0, target_sitk > 0)
    hd = hausdorff_distance_image_filter.GetHausdorffDistance()
    return hd