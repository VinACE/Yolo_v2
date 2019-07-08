# -*- coding:utf-8 -*-

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import io

from torchvision import transforms
from torchsummary.torchsummary import summary
from PIL import Image, ImageDraw
from utilities.utils import visualize_GT
import yolov2
np.set_printoptions(precision=4, suppress=True)


def test(params):

    input_height = params["input_height"]
    input_width = params["input_width"]

    data_path = params["data_path"]
    datalist_path = params["datalist_path"]
    class_path = params["class_path"]
    num_gpus = [i for i in range(1)]
    checkpoint_path = params["checkpoint_path"]
    USE_GTCHECKER = params["use_gtcheck"]


    num_class = params["num_class"]
    num_achors = 5


    with open(class_path) as f:
        class_list = f.read().splitlines()

    anchors = torch.tensor([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053],
               [11.2364, 10.0071]])
    anchor = anchors[:,:] / 13
    anchor = anchor.cpu().data.numpy()
    objness_threshold = 0.1
    class_threshold = 0.1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = yolov2.YOLOv2()
    # model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()
    print("device : ", device)
    if device is "cpu":
        model = torch.nn.DataParallel(net)
    else:
        model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()

    # model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.load_state_dict(torch.load("/home/madhevan/Yolo_v2/ckpt_noHash_ep00100_loss12.2638_lr0.0001.pth.tar"))
    model.eval()
    

    image_path = os.path.join(data_path, "JPEGImages")

    if not (datalist_path =='./'):
        root = next(os.walk(os.path.abspath(data_path)))[0]
        dir = next(os.walk(os.path.abspath(data_path)))[1]
        #files =['000001','000002','000003','000004','000005','000006','000007','000008','000009','000011','000012','000013','000014','000015','000016','000017','000018','000019']
        #files = next(os.walk(os.path.abspath(data_path)))[2]
        files =[]
        with io.open(datalist_path,encoding='utf8') as f:
         for i in f.readlines():
            files.append(i.splitlines()[0])
        for idx in range(len(files)):
            files[idx] += '.jpg'

    else:
        root, dir, files = next(os.walk(os.path.abspath(image_path)))


    for file in files:
        extension = file.split(".")[-1]
        if extension not in ["jpeg", "jpg", "png", "JPEG", "JPG", "PNG"]:
            continue

        img = Image.open(os.path.join(image_path, file)).convert('RGB')

        # PRE-PROCESSING
        input_img = img.resize((input_width, input_height))
        input_img = transforms.ToTensor()(input_img)
        c, w, h = input_img.shape
        input_img = input_img.view(1, c, w, h)
        input_img = input_img.to(device)

        # INVERSE TRANSFORM IMAGE########
        # inverseTimg = transforms.ToPILImage()(input_img)
        W, H = img.size
        draw = ImageDraw.Draw(img)

        dx = W // 13
        dy = H // 13
        ##################################


        # INFERENCE
        outputs = model(input_img)

        # outputs[:,:,:,0:3] = torch.sigmoid(outputs[:,:,:,0:3])
        # outputs[:,:,:,25:28] = torch.sigmoid(outputs[:,:,:,25:28])
        # outputs[:,:,:,50:53] = torch.sigmoid(outputs[:,:,:,50:53])
        # outputs[:,:,:,75:78] = torch.sigmoid(outputs[:,:,:,75:78])
        # outputs[:,:,:,100:103] = torch.sigmoid(outputs[:,:,:,100:103])


        b, w, h, c = outputs.shape
        outputs = outputs.view(w, h, c)
        threshold_map = torch.zeros(13,13,20,dtype=outputs.dtype,device=outputs.device)
        # class_prob = torch.zeros(13,13,5,28,dtype=outputs.dtype,device=outputs.device)
        class_prob = torch.zeros(13,13,25,28,dtype=outputs.dtype,device=outputs.device) ## TODO need to check this for the threshold error...
        pred_bbox = torch.zeros(outputs.shape[0],outputs.shape[1],num_achors,num_class,dtype=outputs.dtype,device=outputs.device)
        final_bbox = nms(outputs,class_prob,class_threshold,num_class,img.size,anchor)

        ###################################################################################
        #mAP 측정
        '''
        test_dataset = VOC(root=data_path, transform=composed, class_path=class_path, datalist_path=datalist_path)

        test_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=1,
                                               shuffle=True,
                                               collate_fn=detection_collate)

        for i, (images, labels, sizes) in enumerate(train_loader):
            # labels.shape = 1 x 13 x 13 x 6
            labels = labels.to(device)

        precision, recall = compute_mAP(final_bbox,gt_bbox)
        '''
        ###################################################################################


        print("IMAGE SIZE")
        print("width : {}, height : {}".format(W, H))
        print("\n\n\n\n")
        #print(outputs[4,7,25:50])
        #print(outputs[6,6,75:100])
        
        for i in range(13):
            for j in range(13):
                draw.rectangle(((dx * i, dy * j), (dx * i + dx, dy * j + dy)), outline='#00ff88')

        #if  one >= objness_threshold:
        for idx in final_bbox:
            i = idx[26]
            j = idx[25]
            anchor_idx = int(idx[27])
            x_start_point = dx * i
            y_start_point = dy * j
            center_x = sigmoid(idx[1]) * dx + x_start_point
            center_y = sigmoid(idx[2]) * dx + y_start_point
            w_ratio = idx[3] 
            h_ratio = idx[4] 
            final_w_ratio = anchor[anchor_idx,0] * np.exp(w_ratio)
            final_h_ratio = anchor[anchor_idx,1] * np.exp(h_ratio)
            # final_w_ratio = final_w_ratio * final_w_ratio
            # final_h_ratio = final_h_ratio * final_h_ratio
            width = int(final_w_ratio * W)
            height = int(final_h_ratio * H)
            #width = int(final_w_ratio * W)
            #height = int(final_w_ratio * H)
            xmin = center_x - (width // 2)
            ymin = center_y - (height // 2)
            xmax = xmin + width
            ymax = ymin + height
            draw.rectangle(((xmin + 2, ymin + 2), (xmax - 2, ymax - 2)), outline="blue")
            #print(np.argmax(idx[5:5+num_class]))
            #print(5 + np.argmax(idx[5:5+num_class]))
            draw.text((xmin + 5, ymin + 5), "{}: {:.2f}".format(class_list[np.argmax(idx[5:5+num_class])], idx[5 + np.argmax(idx[5:5+num_class])]))
            draw.ellipse(((center_x - 2, center_y - 2),
                          (center_x + 2, center_y + 2)),
                         fill='blue')
    
                        # LOG
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.show()
        plt.close()

def nms(outputs,class_prob,class_threshold,num_class,img_size,anchor):
    outputs_np = outputs.cpu().data.numpy()
    final_bbox = []
    for j in range(outputs.shape[0]):
        for i in range(outputs.shape[1]):
            output1 = outputs[i,j,:25]
            output2 = outputs[i,j,25:50]
            output3 = outputs[i,j,50:75]
            output4 = outputs[i,j,75:100]
            output5 = outputs[i,j,100:]

            class_prob[i,j,0,:5] = output1[:5]
            class_prob[i,j,1,:5] = output2[:5]
            class_prob[i,j,2,:5] = output3[:5]
            class_prob[i,j,3,:5] = output4[:5]
            class_prob[i,j,4,:5] = output5[:5]
            # obj_prob x class_prob
            class_prob[i,j,0,5:25] = output1[0] * output1[5:]
            class_prob[i,j,1,5:25] = output2[0] * output2[5:]
            class_prob[i,j,2,5:25] = output3[0] * output3[5:]
            class_prob[i,j,3,5:25] = output4[0] * output4[5:]
            class_prob[i,j,4,5:25] = output5[0] * output5[5:]
            class_prob[i,j,:,25] = i
            class_prob[i,j,:,26] = j
            class_prob[i,j,0,27] = 0
            class_prob[i,j,1,27] = 1
            class_prob[i,j,2,27] = 2
            class_prob[i,j,3,27] = 3
            class_prob[i,j,4,27] = 4

            

    class_prob_view = class_prob.view(-1,28)
    class_prob_view_np = class_prob_view.cpu().data.numpy()

    for i in range(num_class):
        leaderboard = 1
        idx = np.argsort(class_prob_view_np[:,5+ i])[::-1]
        sort_class_prob = class_prob_view_np[idx]
        test_threshold = sort_class_prob[0,i+5]
        if(test_threshold >= class_threshold):
            # import pdb;pdb.set_trace()
            while(sort_class_prob[leaderboard,i+5] >= class_threshold):
                iou = compute_iou(sort_class_prob[0,:],sort_class_prob[leaderboard,:],img_size,anchor)
                if iou < 0.5:
                    final_bbox.append(sort_class_prob[leaderboard,:])
                leaderboard +=1

            # print(i) 
            final_bbox.append(sort_class_prob[0,:])
            
              
            print(final_bbox)     
    return final_bbox
    #output = 13 x 13 x 25 x n

def compute_iou(pred_bbox, true_bbox, img_size, anchor):    
    #truexmin, truexmax, trueymin, trueymax , predboxxmin, predboxxmax, predboxymin, 
    #predboxymax
    w, h = img_size

    pred_achor_idx = int(pred_bbox[27])
    pred_bbox_w = anchor[pred_achor_idx,0] * np.exp(pred_bbox[3])
    pred_bbox_h = anchor[pred_achor_idx,1] * np.exp(pred_bbox[4])

    true_achor_idx = int(true_bbox[27])
    true_bbox_w = anchor[true_achor_idx,0] * np.exp(true_bbox[3])
    true_bbox_h = anchor[true_achor_idx,1] * np.exp(true_bbox[4])

    pred_bbox_c_x = ( pred_bbox[1] + pred_bbox[26] ) * w
    pred_bbox_c_y = ( pred_bbox[2] + pred_bbox[25] ) * h
    pred_bbox_w = pred_bbox_w #* pred_bbox_w
    pred_bbox_h = pred_bbox_h #* pred_bbox_h

    pred_bbox_xmin = pred_bbox_c_x - pred_bbox_w/2
    pred_bbox_xmax = pred_bbox_c_x + pred_bbox_w/2
    pred_bbox_ymin = pred_bbox_c_y - pred_bbox_h/2
    pred_bbox_ymax = pred_bbox_c_y + pred_bbox_h/2

    true_bbox_c_x = ( true_bbox[1] + true_bbox[26] ) * w
    true_bbox_c_y = ( true_bbox[2] + true_bbox[25] ) * h
    true_bbox_w = true_bbox_w #* true_bbox_w
    true_bbox_h = true_bbox_h #* true_bbox_h

    true_bbox_xmin = true_bbox_c_x - true_bbox_w/2
    true_bbox_xmax = true_bbox_c_x + true_bbox_w/2
    true_bbox_ymin = true_bbox_c_y - true_bbox_h/2
    true_bbox_ymax = true_bbox_c_y + true_bbox_h/2

    pred_bbox_area = (pred_bbox_xmax - pred_bbox_xmin + 1) * (pred_bbox_ymax - pred_bbox_ymin + 1)
    true_bbox_area = (true_bbox_xmax - true_bbox_xmin + 1) * (true_bbox_ymax - true_bbox_ymin + 1)
    
    inter_x_min = max(true_bbox_xmin, pred_bbox_xmin)
    inter_y_min = max(true_bbox_ymax, pred_bbox_ymax)        
    inter_x_max = min(true_bbox_xmax, pred_bbox_xmax)        
    inter_y_max = min(true_bbox_ymax, pred_bbox_ymax)         

    inter_area = max(0,inter_x_max - inter_x_min + 1) * max(0,inter_y_max - inter_y_min + 1)

    iou = inter_area / float(pred_bbox_area + true_bbox_area - inter_area)

    return iou
def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))