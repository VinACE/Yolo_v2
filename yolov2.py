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
            #nn.Linear(13 * 13 * 25 * 5, 13 * 13 * 25 * 5)
        )

    def forward(self, x):

        out = self.backbone(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(-1,13,13,125)

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
    anchors = torch.tensor([[1.08,1.19],[3.42,4.41],[6.63,11.38],[9.42,5.11],[16.62,10.52]],dtype=target.dtype,device=device)
    anchors = anchors / 13 #image 기준으로 나누기
    # output tensor slice
    # output tensor shape is [batch, 13, 13, anchers * (5 + classes) ]
    predict = torch.zeros(output.size(),dtype=output.dtype,device=device)
    predict.data = output.clone()
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
    

    class_output = target[:, :, :, 5] # 10 = bounding_boxes
    class_label = one_hot(target,non_zero, device) # 5 = bounding_boxes
    non_zero_np = non_zero.cpu().data.numpy()

    # predict & obj
    for i in range(num_object):
        #sigmoid + Cx, Cy
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],0] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],0]) #+ non_zero[i,1].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],1] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],1]) #+ non_zero[i,1].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],2] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],2]) #+ non_zero[i,2].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],26] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],26]) #+ non_zero[i,1].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],27] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],27]) #+ non_zero[i,2].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],51] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],51]) #+ non_zero[i,1].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],52] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],52]) #+ non_zero[i,2].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],76] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],76]) #+ non_zero[i,1].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],77] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],77]) #+ non_zero[i,2].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],101] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],101]) #+ non_zero[i,1].float()
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],102] = torch.sigmoid(output[non_zero[i,0],non_zero[i,1],non_zero[i,2],102]) #+ non_zero[i,2].float()
        #bw, bh
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],3] = anchors[i,0] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],3] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],4] = anchors[i,1] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],4] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],28] = anchors[i,0] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],28] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],29] = anchors[i,1] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],29] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],53] = anchors[i,0] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],53] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],54] = anchors[i,1] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],54] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],78] = anchors[i,0] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],78] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],79] = anchors[i,1] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],79] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],103] = anchors[i,0] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],103] )
        predict[non_zero[i,0],non_zero[i,1],non_zero[i,2],104] = anchors[i,1] * torch.exp ( output[non_zero[i,0],non_zero[i,1],non_zero[i,2],104] )
        #ratio
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],:] = 1.
        '''
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],25] = 1
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],50] = 1
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],75] = 1
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],100] = 1
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],5:25] = 1
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],30:50] = 1
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],55:75] = 1
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],80:100] = 1
        ratio[non_zero[i,0],non_zero[i,1],non_zero[i,2],105:125] = 1
        '''


    target_complete_onehot = torch.cat((target[:,:,:,:5],class_label),3)
    target_complete_onehot[:,:,:,3:5] = torch.sqrt(target_complete_onehot[:,:,:,3:5])



    label = torch.cat((target_complete_onehot,target_complete_onehot,target_complete_onehot,target_complete_onehot,target_complete_onehot),3)


    dX = 416 / 13
    dY = 416 / 13
    '''
    iou 계산하는 식
    for i in range(num_object):
        #anchor = (c_x,c_y,width1~5,height1~5)
        #true = same
        anchor = torch.zeros(12,dtype=output.dtype,device=device)
        anchor[0] = non_zero[i,1]
        anchor[1] = non_zero[i,2]
        anchor[2:7] = anchors[:,0]
        anchor[7:] = anchors[:,1]
        
        true = torch.zeros(4,dtype=output.dtype,device=device)
        true[0] = target[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],1] #c_x
        true[1] = target[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],2] #c_y
        true[2] = target[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],3] #w
        true[3] = target[non_zero_np[i,0],non_zero_np[i,1],non_zero_np[i,2],4] #h

        anchor5_iou = compute_iou(anchor,true)

        argmax_anchor_idx = torch.argmax(anchor5_iou)
        pass
    '''

    dog_target = label[non_zero[0,0],non_zero[0,1],non_zero[0,2],:]
    human_target = label[non_zero[1,0],non_zero[1,1],non_zero[1,2],:]
    dog_ratio = ratio[non_zero[0,0],non_zero[0,1],non_zero[0,2],:]
    
    loss = ratio * (label - predict) * (label - predict)

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
    #TODO
    inter_bbox = torch.tensor(10,dtype=target.dtype,device='cuda')
    inter_x_min = torch.max(true_x_min, anchor_x_min)
    inter_y_min = torch.max(true_y_min, anchor_y_min)        
    inter_x_max = torch.min(true_x_max, anchor_x_max)        
    inter_y_max = torch.min(true_y_max, anchor_y_max)       

    inter_area = torch.max(torch.zeros(1,dtype=inter_x_max.dtype,device='cuda'), (inter_x_max - inter_x_min + 1)) * torch.max(torch.zeros(1,dtype=inter_x_max.dtype,device='cuda'),(inter_y_max - inter_y_min + 1))  

    #inter_area = max(0,inter_x_max - inter_x_min + 1) * max(0,inter_y_max - inter_y_min + 1)

    iou = inter_area / (anch_bbox_area + true_bbox_area - inter_area)

    return iou
