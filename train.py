# -*- coding:utf-8 -*-

import os
import warnings

import git
import torch
import torchvision.transforms as transforms

import yolov2
import visdom
from yolov2 import detection_loss_4_yolo
from torchsummary.torchsummary import summary
from utilities.dataloader import detection_collate
from utilities.dataloader import VOC

from utilities.utils import save_checkpoint
from utilities.utils import create_vis_plot
from utilities.utils import update_vis_plot
from utilities.utils import visualize_GT
from utilities.augmentation import Augmenter
from imgaug import augmenters as iaa


warnings.filterwarnings("ignore")
# plt.ion()   # interactive mode
# model = torch.nn.DataParallel(net, device_ids=[0]).cuda()


def train(params):

    # future work variable
    dataset = params["dataset"]
    input_height = params["input_height"]
    input_width = params["input_width"]

    data_path = params["data_path"]
    val_data_path = params["val_data_path"]
    val_datalist_path = params["val_datalist_path"]
    datalist_path = params["datalist_path"]
    class_path = params["class_path"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    learning_rate = params["lr"]
    checkpoint_path = params["checkpoint_path"]

    USE_AUGMENTATION = params["use_augmentation"]
    USE_GTCHECKER = params["use_gtcheck"]
    USE_VISDOM = params["use_visdom"]

    USE_GITHASH = params["use_githash"]
    num_class = params["num_class"]
    num_gpus = [i for i in range(1)]
    with open(class_path) as f:
        class_list = f.read().splitlines()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if (USE_GITHASH):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        short_sha = repo.git.rev_parse(sha, short=7)

    if USE_VISDOM:
        viz = visdom.Visdom(use_incoming_socket=False)
        vis_title = 'YOLOv2'
        vis_legend_Train = ['Train Loss']
        vis_legend_Val = ['Val Loss']
        iter_plot = create_vis_plot(viz, 'Iteration', 'Total Loss', vis_title, vis_legend_Train)
        val_plot = create_vis_plot(viz, 'Iteration', 'Validation Loss', vis_title, vis_legend_Val)

    # 2. Data augmentation setting
    if (USE_AUGMENTATION):
        seq = iaa.SomeOf(2, [
            iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
            iaa.Affine(
                translate_px={"x": 3, "y": 10},
                scale=(0.9, 0.9)
            ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            iaa.AdditiveGaussianNoise(scale=0.1 * 255),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            iaa.Affine(rotate=45),
            iaa.Sharpen(alpha=0.5)
        ])
    else:
        seq = iaa.Sequential([])

    composed = transforms.Compose([Augmenter(seq)])

    # 3. Load Dataset
    # composed
    # transforms.ToTensor
    #TODO : Datalist가 있을때 VOC parsing
    # import pdb;pdb.set_trace()
    train_dataset = VOC(root=data_path,transform=composed, class_path=class_path, datalist_path=datalist_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=detection_collate)
    val_dataset = VOC(root=val_data_path,transform=composed,  class_path=class_path, datalist_path=val_datalist_path)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=detection_collate)
    # 5. Load YOLOv2
    net = yolov2.YOLOv2()
    model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()

    print("device : ", device)
    if device.type == 'cpu':
        model = torch.nn.DataParallel(net)
    else:
        model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()


    # 7.Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Train the model
    total_step = len(train_loader)

    total_train_step = num_epochs * total_step

    # for epoch in range(num_epochs):
    for epoch in range(1, num_epochs + 1):
        train_loss =0
        total_val_loss = 0

        train_total_conf_loss = 0
        train_total_xy_loss = 0
        train_total_wh_loss = 0
        train_total_c_loss = 0
        
        
        val_total_conf_loss = 0
        val_total_xy_loss = 0
        val_total_wh_loss = 0
        val_total_c_loss = 0

        if(epoch %500 ==0 and epoch <1000):
            learning_rate /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        if (epoch == 200) or (epoch == 400) or (epoch == 600) or (epoch == 20000) or (epoch == 30000):
            scheduler.step()
        model.train()
        for i, (images, labels, sizes) in enumerate(train_loader):

            current_train_step = (epoch) * total_step + (i + 1)

            if USE_GTCHECKER:
                visualize_GT(images, labels, class_list)

            images = images.to(device)
            labels = labels.to(device)
            
            dog = labels[0,4,7,:]
            human = labels[0,6,6,:]
            # Forward pass
            outputs = model(images)

            # Calc Loss
            one_loss,conf_loss,xy_loss,wh_loss,class_loss    = detection_loss_4_yolo(outputs, labels, device.type)
            # objness1_loss = detection_loss_4_yolo(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            one_loss.backward()
            optimizer.step()
            train_loss += one_loss.item()
            train_total_conf_loss += conf_loss.item()
            train_total_xy_loss += xy_loss.item()
            train_total_wh_loss += wh_loss.item()
            train_total_c_loss += class_loss.item()
        
        train_total_conf_loss = train_total_conf_loss / len(train_loader)
        train_total_xy_loss= train_total_xy_loss / len(train_loader)
        train_total_wh_loss = train_total_wh_loss /len(train_loader)
        train_total_c_loss = train_total_c_loss /len(train_loader)
        train_epoch_loss = train_loss / len(train_loader)
        update_vis_plot(viz, epoch + 1, train_epoch_loss, iter_plot, None, 'append')

        model.eval()
        with torch.no_grad():

            for j, (v_images, v_labels, v_sizes) in enumerate(val_loader):
                v_images = v_images.to(device)
                v_labels = v_labels.to(device)
                # Forward pass
                v_outputs = model(v_images)
    
                # Calc Loss
                val_loss,conf_loss,xy_loss,wh_loss,class_loss    = detection_loss_4_yolo(v_outputs, v_labels, device.type)
                total_val_loss += val_loss.item()
                val_total_conf_loss += conf_loss.item()
                val_total_xy_loss += xy_loss.item()
                val_total_wh_loss += wh_loss.item()
                val_total_c_loss += class_loss.item()


            val_epoch_loss = total_val_loss / len(val_loader)
            val_total_conf_loss = val_total_conf_loss / len(val_loader)
            val_total_xy_loss= val_total_xy_loss / len(val_loader)
            val_total_wh_loss = val_total_wh_loss /len(val_loader)
            val_total_c_loss = val_total_c_loss /len(val_loader)
            update_vis_plot(viz, epoch + 1, val_epoch_loss, val_plot, None, 'append')

        if (((current_train_step) % 100) == 0) or (current_train_step % 1 == 0 and current_train_step < 300):
            print(
                'epoch: [{}/{}], total step: [{}/{}], batch step [{}/{}], lr: {},one_loss: {:.4f},val_loss: {:.4f}'
                .format(epoch + 1, num_epochs, current_train_step, total_train_step, i + 1, total_step,
                        ([param_group['lr'] for param_group in optimizer.param_groups])[0],
                         one_loss,val_loss ))

        print('train loss',train_epoch_loss,'val loss',val_epoch_loss)
        print('train conf loss',train_total_conf_loss,'val conf loss',val_total_conf_loss)
        print('train xy loss',train_total_xy_loss,'val xy loss',val_total_xy_loss)
        print('train wh loss',train_total_wh_loss,'val wh loss',val_total_wh_loss)
        print('train class loss',train_total_c_loss,'val class loss',val_total_c_loss)
        if not USE_GITHASH:
            short_sha = 'noHash'

        # if ((epoch % 1000) == 0) and (epoch != 0):
        # if ((epoch % 100) == 0) :
        if ((epoch % 10) == 0) :
        #if (one_loss <= 1) :
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "YOLOv2",
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, filename=os.path.join(checkpoint_path, 'ckpt_{}_ep{:05d}_loss{:.04f}_lr{}.pth.tar'.format(short_sha, epoch, one_loss.item(), ([param_group['lr'] for param_group in optimizer.param_groups])[0])))
            # print(dir(model))
            filename = os.path.join(checkpoint_path, 'ckpt_{}_ep{:05d}_loss{:.04f}_lr{}.pth.tar'.format(short_sha, epoch, one_loss.item(), ([param_group['lr'] for param_group in optimizer.param_groups])[0]))
            torch.save(model.module.state_dict(),filename)
