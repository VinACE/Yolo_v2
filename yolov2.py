# -*- coding:utf-8 -*-

import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary
from utilities import dataloader
from dataloader import VOC
from model.darknet import Darknet19
import numpy as np
import matplotlib.pyplot as plt

import visdom

class YOLOv2(nn.Module):
    def __init__(self):
        super(YOLOv2, self).__init__()
        
        self.darknet = Darknet19()
        self.backbone = self.darknet.features
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024,1024, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1024,1024, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(1024,1024, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1024, 25 * 5, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.Linear(13 * 13 * 25 * 5, 13 * 13 * 25 * 5)
        )

    def forward(self, x):

        out = self.backbone(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(-1,13,13,125)
        out[:,:,:,0] = torch.sigmoid(out[:,:,:,0])
        out[:,:,:,5:25] = torch.sigmoid(out[:,:,:,5:25])
        out[:,:,:,25] = torch.sigmoid(out[:,:,:,25])
        out[:,:,:,30:50] = torch.sigmoid(out[:,:,:,30:50])
        out[:,:,:,50] = torch.sigmoid(out[:,:,:,50])
        out[:,:,:,55:75] = torch.sigmoid(out[:,:,:,55:75])
        out[:,:,:,75] = torch.sigmoid(out[:,:,:,75])
        out[:,:,:,80:100] = torch.sigmoid(out[:,:,:,80:100])
        out[:,:,:,100] = torch.sigmoid(out[:,:,:,100])
        out[:,:,:,105:125] = torch.sigmoid(out[:,:,:,105:125])

        return out

    def unfreeze_darknet(self,model):
        #self.backbone 을 품 
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
    #anchors
    anchors = torch.tensor([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053],
               [11.2364, 10.0071]],dtype=target.dtype,device=device)
    anchors = anchors / 13 #image 기준으로 나누기
    # output tensor slice
    # output tensor shape is [batch, 13, 13, anchers * (5 + classes) ]
    predict = torch.zeros(output.size(),dtype=output.dtype,device=device)

    label = torch.zeros(output.size(),dtype=output.dtype,device=device)
    #obj
    non_zero = (target[:, :, :, 0]==1).nonzero()
    num_object,_ = non_zero.shape


    #target & setting no obj
    ratio = torch.zeros(output.size(),dtype=output.dtype,device=device)
    ratio[:,:,:,0] = lambda_noobj
    ratio[:,:,:,25] = lambda_noobj
    ratio[:,:,:,50] = lambda_noobj
    ratio[:,:,:,75] = lambda_noobj
    ratio[:,:,:,100] = lambda_noobj
    
    #convert target one_hot
    class_output = target[:, :, :, 5] # 10 = bounding_boxes
    class_label = one_hot(target,non_zero, device) # 5 = bounding_boxes
    non_zero_np = non_zero.cpu().data.numpy()
    target_complete_onehot = torch.cat((target[:,:,:,:5],class_label),3)






    


    dX = 416 / 13
    dY = 416 / 13
    #iou 계산하는 식, obj 있을 때
    for i in range(num_object):
        #anchor = (c_x,c_y,width1~5,height1~5)
        #true = same
        anchor = torch.zeros(12,dtype=output.dtype,device=device)
        anchor[0] = non_zero[i,1]
        anchor[1] = non_zero[i,2]
        anchor[2:7] = anchors[:,0]
        anchor[7:] = anchors[:,1]
        
        true = torch.zeros(4,dtype=output.dtype,device=device)
        true = target[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],1:5] #c_x,c_y,w,h

        anchor5_iou = compute_iou(anchor,true)

        argmax_anchor_idx = torch.argmax(anchor5_iou)

        #target
        label[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx : 25 * argmax_anchor_idx + 25 ] = target_complete_onehot[non_zero[i,0],non_zero[i,1],non_zero[i,2],:]
        

        #predict 
        output[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 1 : 25 * argmax_anchor_idx + 3 ] = \
            torch.sigmoid( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 1 : 25 * argmax_anchor_idx + 3 ] )
        output[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 3] = \
            anchors[argmax_anchor_idx,0] * torch.exp( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 3] )
        output[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 4] = \
            anchors[argmax_anchor_idx,1] * torch.exp( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 4] )

        # output[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 5 : 25 * argmax_anchor_idx + 25] = \
        #     torch.sigmoid( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 5 : 25 * argmax_anchor_idx + 25 ] )

        
        
        #ratio
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx] = 1.
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 1 : 25 * argmax_anchor_idx +5] = 5.
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],25 * argmax_anchor_idx + 5 : 25 * argmax_anchor_idx + 25 ] = 1.
        # print(label[0,4,7,25:50])
        # print(label[0,6,6,100:])

    loss = ratio * (label - output) * (label - output)
    loss = loss.view(-1)
    loss = torch.sum(loss) / b


    return loss

def compute_iou(anchor, target):    
    
    anchor_x_min = anchor[0] * 416 / 13 - anchor[2:7] / 2 * 416
    anchor_x_max = anchor[0] * 416 / 13 + anchor[2:7] / 2 * 416
    anchor_y_min = anchor[1] * 416 / 13 - anchor[7:] / 2 * 416
    anchor_y_max = anchor[1] * 416 / 13 + anchor[7:] / 2 * 416

    anch_bbox_area = (anchor_x_max - anchor_x_min + 1) * (anchor_y_max - anchor_y_min + 1)

    true_x_min = ( anchor[0] + target[0] ) * 416 / 13 - target[2] / 2 * 416
    true_x_max = ( anchor[0] + target[0] ) * 416 / 13 + target[2] / 2 * 416
    true_y_min = ( anchor[1] + target[1] ) * 416 / 13 - target[3] / 2 * 416
    true_y_max = ( anchor[1] + target[1] ) * 416 / 13 + target[3] / 2 * 416

    true_bbox_area = (true_x_max - true_x_min + 1) * (true_y_max - true_y_min + 1)

    inter_x_min = torch.max(true_x_min, anchor_x_min)
    inter_y_min = torch.max(true_y_min, anchor_y_min)        
    inter_x_max = torch.min(true_x_max, anchor_x_max)        
    inter_y_max = torch.min(true_y_max, anchor_y_max)       

    inter_area = torch.max(torch.zeros(1,dtype=inter_x_max.dtype,device='cuda'), (inter_x_max - inter_x_min + 1)) * torch.max(torch.zeros(1,dtype=inter_x_max.dtype,device='cuda'),(inter_y_max - inter_y_min + 1))  

    #inter_area = max(0,inter_x_max - inter_x_min + 1) * max(0,inter_y_max - inter_y_min + 1)

    iou = inter_area / (anch_bbox_area + true_bbox_area - inter_area)

    return iou
