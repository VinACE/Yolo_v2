# -*- coding:utf-8 -*-

import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary
from utilities import dataloader
from dataloader import VOC
from model import Darknet19
import numpy as np
import matplotlib.pyplot as plt

import visdom

class YOLOv2(BaseModel):
    def __init__(self):
        super(YOLOv2, self).__init__()
        
        self.darknet = Darknet19()
        self.backbone = self.darknet.features
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024,1024, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1024,1024, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(1024,1024, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1024, 25 * 5, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Linear()
        )

    def forward(self, x):

        out = self.backbone(x)
        out = self.freeze_darknet(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)


        return out
    def freeze_darknet(self,model):
        #self.backbone 을 얼림 
        model.eval()
        for params in model.parameters():
            params.requires_grad = False

    def unfreeze_darknet(self,model):
        #self.backbone 을 품 
        model.train()
        for params in model.parameters():
            params.requires_grad = True


# def detection_loss_4_yolo(output, target):
def detection_loss_4_yolo(output, target, device):
    from utilities.utils import one_hot

    # hyper parameter

    lambda_coord = 5
    lambda_noobj = 0.5

    # check batch size
    b, _, _, _ = target.shape
    _, _, _, n = output.shape

    # output tensor slice
    # output tensor shape is [batch, 7, 7, bounding_boxes + classes]
    objness1_output = output[:, :, :, 0]
    x_offset1_output = output[:, :, :, 1]
    y_offset1_output = output[:, :, :, 2]
    width_ratio1_output = output[:, :, :, 3]
    height_ratio1_output = output[:, :, :, 4]
    objness2_output = output[:, :, :, 5]
    x_offset2_output = output[:, :, :, 6]
    y_offset2_output = output[:, :, :, 7]
    width_ratio2_output = output[:, :, :, 8]
    height_ratio2_output = output[:, :, :, 9]


    pred_bbox = output[:, :, :, :9]
    class_output = output[:, :, :, 10:] # 10 = bounding_boxes
    num_cls = class_output.shape[-1]
    non_zero = (target[:, :, :, 0]==1).nonzero()

    true_bbox = target[:, :, :, :5]
    class_label = one_hot(class_output, target[:, :, :, 5],non_zero, device) # 5 = bounding_boxes

    no_obj_bbox = torch.zeros(1,5,dtype=true_bbox.dtype,device=device)
    label = torch.zeros(output.size(),dtype=output.dtype,device=device) #이미 no_obj_bbox 세팅 되어있음
    label[:, :, :, 10:] = class_label
    ratio = torch.zeros(output.size(),dtype=output.dtype,device=device)
    ratio_nobj = torch.tensor([[lambda_noobj,0,0,0,0]],dtype=true_bbox.dtype,device=device)
    ratio[:, :, :, :5] = ratio_nobj
    ratio[:, :, :, 5:10] = ratio_nobj
    ratio_obj = torch.tensor([[1,lambda_coord,lambda_coord,lambda_coord,lambda_coord]],dtype=true_bbox.dtype,device=device)
    ratio_cls = torch.ones(20,dtype=true_bbox.dtype,device=device)

    dog_label = target[0,2,4,:]
    human_label = target[0,3,3,:]
    #object is exist
    '''
    pred_bbox1 : coord_obj의 예측값1
    pred_bbox2 : coord_obj의 예측값2
    shape = [1,5]
    '''
    pred_bbox1 = output[non_zero[:,0],non_zero[:,1],non_zero[:,2], 0:5]
    pred_bbox2 = output[non_zero[:,0],non_zero[:,1],non_zero[:,2], 5:10]
    coor_true_bbox = true_bbox[non_zero[:,0],non_zero[:,1],non_zero[:,2], :]

    pred_bbox1_np = pred_bbox1.cpu().data.numpy()
    pred_bbox2_np = pred_bbox2.cpu().data.numpy()
    coor_true_bbox_np = coor_true_bbox.cpu().data.numpy()
    non_zero_np = non_zero.cpu().data.numpy()

    num_object, _ = coor_true_bbox.shape

    for i in range(num_object):
        
        pred_bbox1_center_x = ( non_zero_np[i,1] + pred_bbox1_np[i,1] )*448 // 7 
        pred_bbox1_center_y = ( non_zero_np[i,2] + pred_bbox1_np[i,2] )*448 // 7
        pred_bbox1_x_min =  pred_bbox1_center_x - ( 448*pred_bbox1_np[i,3] // 2 )
        pred_bbox1_x_max =  pred_bbox1_center_x + ( 448*pred_bbox1_np[i,3] // 2 )
        pred_bbox1_y_min =  pred_bbox1_center_y - ( 448*pred_bbox1_np[i,4] // 2 )
        pred_bbox1_y_max =  pred_bbox1_center_y + ( 448*pred_bbox1_np[i,4] // 2 )

        pred_bbox2_center_x = ( non_zero_np[i,1] + pred_bbox2_np[i,1] )*448 // 7 
        pred_bbox2_center_y = ( non_zero_np[i,2] + pred_bbox2_np[i,2] )*448 // 7
        pred_bbox2_x_min =  pred_bbox2_center_x - ( 448*pred_bbox2_np[i,3] // 2 )
        pred_bbox2_x_max =  pred_bbox2_center_x + ( 448*pred_bbox2_np[i,3] // 2 )
        pred_bbox2_y_min =  pred_bbox2_center_y - ( 448*pred_bbox2_np[i,4] // 2 )
        pred_bbox2_y_max =  pred_bbox2_center_y + ( 448*pred_bbox2_np[i,4] // 2 )

        coor_tbbox_center_x = ( non_zero_np[i,1] + coor_true_bbox_np[i,1] )*448 // 7 
        coor_tbbox_center_y = ( non_zero_np[i,2] + coor_true_bbox_np[i,2] )*448 // 7
        coor_tbbox_x_min =  coor_tbbox_center_x - ( 448*coor_true_bbox_np[i,3] // 2 )
        coor_tbbox_x_max =  coor_tbbox_center_x + ( 448*coor_true_bbox_np[i,3] // 2 )
        coor_tbbox_y_min =  coor_tbbox_center_y - ( 448*coor_true_bbox_np[i,4] // 2 )
        coor_tbbox_y_max =  coor_tbbox_center_y + ( 448*coor_true_bbox_np[i,4] // 2 )

        if(compute_iou( coor_tbbox_x_min, coor_tbbox_x_max, coor_tbbox_y_min, coor_tbbox_y_max, 
                           pred_bbox1_x_min, pred_bbox1_x_max, pred_bbox1_y_min, pred_bbox1_y_max) >=\
                            compute_iou( coor_tbbox_x_min, coor_tbbox_x_max, coor_tbbox_y_min, coor_tbbox_y_max, 
                           pred_bbox2_x_min, pred_bbox2_x_max, pred_bbox2_y_min, pred_bbox2_y_max)):
            label[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],:5] = target[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],:5]
            ratio[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],:5] = ratio_obj
            ratio[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],5:10] = ratio_nobj
            ratio[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],10:] = ratio_cls
        else:
            label[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],5:10] = target[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],:5]
            ratio[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],:5] = ratio_nobj
            ratio[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],5:10] = ratio_obj
            ratio[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],10:] = ratio_cls


    
    label[:,:,:,3:5] = torch.sqrt(label[:,:,:,3:5])
    label[:,:,:,8:10] = torch.sqrt(label[:,:,:,8:10])
    loss = ratio * (label - output) * (label - output)
    loss = loss.view(-1)
    loss = torch.sum(loss) / b
    # label tensor slice
    objness_label1 = label[:, :, :, 0]
    x_offset_label1 = label[:, :, :, 1]
    y_offset_label1 = label[:, :, :, 2]
    width_ratio_label1 = label[:, :, :, 3]
    height_ratio_label1 = label[:, :, :, 4]

    objness_label2 = label[:, :, :, 5]
    x_offset_label2 = label[:, :, :, 6]
    y_offset_label2 = label[:, :, :, 7]
    width_ratio_label2 = label[:, :, :, 8]
    height_ratio_label2 = label[:, :, :, 9]

    # ratio tensor slice
    objness_ratio1 = ratio[:, :, :, 0]
    offset_width_ratio1 = ratio[:, :, :, 1]
    objness_ratio2 = ratio[:, :, :, 5]
    offset_width_ratio2 = ratio[:, :, :, 6]

    noobjness_label1 = torch.neg(torch.add(objness_label1, -1))
    noobjness_label2 = torch.neg(torch.add(objness_label2, -1))

    obj_coord_loss1 = torch.sum(offset_width_ratio1 * \
                      (objness_label1 *(torch.pow(x_offset1_output - x_offset_label1, 2) +
                                    torch.pow(y_offset1_output - y_offset_label1, 2))))

    obj_coord_loss2 = torch.sum(offset_width_ratio2 * \
                      (objness_label2 *
                        (torch.pow(x_offset2_output - x_offset_label2, 2) +
                                    torch.pow(y_offset2_output - y_offset_label2, 2))))

    obj_coord_loss = obj_coord_loss1 + obj_coord_loss2      

    obj_size_loss1 = torch.sum(offset_width_ratio1 * \
                     (objness_label1 *
                               (torch.pow((width_ratio1_output - torch.sqrt(width_ratio_label1)), 2) +
                                torch.pow((height_ratio1_output -  torch.sqrt(height_ratio_label1)), 2))))

    obj_size_loss2 = torch.sum(offset_width_ratio2 * \
                     (objness_label2 *
                               (torch.pow((width_ratio2_output - torch.sqrt(width_ratio_label2)), 2) +
                                torch.pow((height_ratio2_output - torch.sqrt(height_ratio_label2)), 2))))

    obj_size_loss = obj_size_loss1 + obj_size_loss2

    no_obj_label1 = torch.neg(torch.add(objness1_output, -1))
    no_obj_label2 = torch.neg(torch.add(objness2_output, -1))

    noobjness1_loss = torch.sum(objness_ratio1 * no_obj_label1 * torch.pow(objness1_output - objness_label1, 2))
    noobjness2_loss = torch.sum(objness_ratio2 * no_obj_label2 * torch.pow(objness2_output - objness_label2, 2))

    noobjness_loss = noobjness1_loss + noobjness2_loss

    objness_loss = torch.sum(objness_ratio1 * torch.pow(objness1_output - objness_label1, 2) + objness_ratio2 * torch.pow(objness2_output - objness_label2, 2))

    objectness_cls_map = target[:,:,:,0].unsqueeze(-1)

    for i in range(num_cls - 1):
        objectness_cls_map = torch.cat((objectness_cls_map, target[:,:,:,0].unsqueeze(-1)), 3)

    obj_class_loss = torch.sum(objectness_cls_map * torch.pow(class_output - class_label, 2))

    total_loss = (obj_coord_loss + obj_size_loss + noobjness_loss +  objness_loss + obj_class_loss)
    total_loss = total_loss / b


    return total_loss, obj_coord_loss / b, obj_size_loss / b, noobjness_loss / b, obj_class_loss / b, objness_loss / b, loss

def compute_iou(truexmin, truexmax, trueymin, trueymax , predboxxmin, predboxxmax, predboxymin, predboxymax):    
    
    pred_bbox_area = (predboxxmax - predboxxmin + 1) * (predboxymax - predboxymin + 1)
    true_bbox_area = (truexmax - truexmin + 1) * (trueymax - trueymin + 1)
    
    inter_x_min = max(truexmin, predboxxmin)
    inter_y_min = max(trueymin, predboxymin)        
    inter_x_max = min(truexmax, predboxxmax)        
    inter_y_max = min(trueymax, predboxymax)         

    inter_area = max(0,inter_x_max - inter_x_min + 1) * max(0,inter_y_max - inter_y_min + 1)

    iou = inter_area / float(pred_bbox_area + true_bbox_area - inter_area)

    return iou
