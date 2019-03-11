# -*- coding:utf-8 -*-

import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary
from utilities import dataloader
from dataloader import VOC

import numpy as np
import matplotlib.pyplot as plt

import visdom


class YOLOv1(nn.Module):
    def __init__(self, params):

        self.dropout_prop = params["dropout"]
        self.num_classes = params["num_class"]
        self.bounding_boxes = 10
        super(YOLOv1, self).__init__()
        # LAYER 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 4
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer14 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 5
        self.layer17 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer19 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer20 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer21 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer22 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())

        # LAYER 6
        self.layer23 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer24 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 256, 512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prop)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 7 * 7 * ((self.bounding_boxes) + self.num_classes))
        )

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out = self.layer19(out)
        out = self.layer20(out)
        out = self.layer21(out)
        out = self.layer22(out)
        out = self.layer23(out)
        out = self.layer24(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.reshape((-1, 7, 7, ((self.bounding_boxes) + self.num_classes)))
        out[:, :, :, 0] = torch.sigmoid(out[:, :, :, 0])  # sigmoid to objness1_output
        out[:, :, :, 5] = torch.sigmoid(out[:, :, :, 5])  # sigmoid to objness2_output
        out[:, :, :, self.bounding_boxes:] = torch.sigmoid(out[:, :, :, self.bounding_boxes:])  # sigmoid to class_output

        return out


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
