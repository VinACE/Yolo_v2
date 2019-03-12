# -*- coding:utf-8 -*-

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import io

from torchvision import transforms
from torchsummary.torchsummary import summary
from PIL import Image, ImageDraw

np.set_printoptions(precision=4, suppress=True)


def test(params):

    input_height = params["input_height"]
    input_width = params["input_width"]

    data_path = params["data_path"]
    datalist_path = params["datalist_path"]
    class_path = params["class_path"]
    num_gpus = [i for i in range(1)]
    checkpoint_path = params["checkpoint_path"]

    USE_SUMMARY = params["use_summary"]

    num_class = params["num_class"]

    with open(class_path) as f:
        class_list = f.read().splitlines()

    objness_threshold = 0.5
    class_threshold = 0.3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = yolov2.YOLOv2()
    # model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()
    print("device : ", device)
    if device is "cpu":
        model = torch.nn.DataParallel(net)
    else:
        model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()

    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()

    if USE_SUMMARY:
        summary(model, (3, 448, 448))

    image_path = os.path.join(data_path, "JPEGImages")

    if not (datalist_path =='./'):
        root = next(os.walk(os.path.abspath(data_path)))[0]
        dir = next(os.walk(os.path.abspath(data_path)))[1]
        files =['000001']#,'000002','000003','000004']#,'000005','000006','000007','000008','000009',]
        #with io.open(datalist_path,encoding='utf8') as f:
        #    for i in f.readlines():
        #        files.append(i.splitlines()[0])
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

        # INVERSE TRANSFORM IMAGE########
        # inverseTimg = transforms.ToPILImage()(input_img)
        W, H = img.size
        draw = ImageDraw.Draw(img)

        dx = W // 13
        dy = H // 13
        ##################################

        input_img = input_img.view(1, c, w, h)
        input_img = input_img.to(device)

        # INFERENCE
        outputs = model(input_img)
        b, w, h, c = outputs.shape
        outputs = outputs.view(w, h, c)
        outputs_np = outputs.cpu().data.numpy()
        print("outputs_np : {}".format(outputs_np.shape))

        objness1 = outputs[:, :, 0].unsqueeze(-1).cpu().data.numpy()
        objness2 = outputs[:, :, 5].unsqueeze(-1).cpu().data.numpy()
        
        cls_map = outputs[:, :, 10:].cpu().data.numpy()


        threshold_map1 = np.multiply(objness1, cls_map)
        threshold_map2 = np.multiply(objness2, cls_map)
        threshold_map = np.stack([threshold_map1,threshold_map2],axis=0)

        objness = np.stack([objness1,objness2],axis=2)
        objness = objness.reshape(7, 7, 2)

        #print("obj1 : ",objness1[:][:])
        #print("obj2 : ",objness2[:][:])
        #print("cls :",cls_map[:][:])

        print("OBJECTNESS : {}".format(objness.shape))
        print("\n\n\n")
        print("CLS MAP : {}".format(cls_map.shape))
        print(cls_map[0])
        print("\n\n\n")
        print("MULTIPLICATION : {}".format(threshold_map.shape))
        print(threshold_map[0, :, :,14],threshold_map[0, :, :,11])
        print("\n\n\n")

        print("IMAGE SIZE")
        print("width : {}, height : {}".format(W, H))
        print("\n\n\n\n")
        print("outputs_np : {}".format(outputs_np.shape))

        try:
            for i in range(13):
                for j in range(13):
                    draw.rectangle(((dx * i, dy * j), (dx * i + dx, dy * j + dy)), outline='#00ff88')
                    one = objness[i][j][0]
                    two = objness[i][j][1]
                    one_block = outputs_np[i][j][:5]
                    two_block = outputs_np[i][j][5:10]
                    class_block = outputs_np[i][j][10:]
                    one_clsprob = class_block * one
                    two_clsprob = class_block * two
                    one_cls_idx = np.argmax(one_clsprob)
                    two_cls_idx = np.argmax(two_clsprob)

                    if  one >= objness_threshold:
                        x_start_point = dx * i
                        y_start_point = dy * j
                        x_shift = one_block[1]
                        y_shift = one_block[2]
                        center_x = int((one_block[1] * W / 7.0) + (i * W / 7.0))
                        center_y = int((one_block[2] * H / 7.0) + (j * H / 7.0))
                        w_ratio = one_block[3]
                        h_ratio = one_block[4]
                        w_ratio = w_ratio * w_ratio
                        h_ratio = h_ratio * h_ratio
                        width = int(w_ratio * W)
                        height = int(h_ratio * H)
                        xmin = center_x - (width // 2)
                        ymin = center_y - (height // 2)
                        xmax = xmin + width
                        ymax = ymin + height
                        if one_clsprob[one_cls_idx] > class_threshold:
                            if (one_cls_idx == two_cls_idx) and (one_clsprob[one_cls_idx] >= two_clsprob[two_cls_idx]):
                                draw.rectangle(((xmin + 2, ymin + 2), (xmax - 2, ymax - 2)), outline="blue")
                                draw.text((xmin + 5, ymin + 5), "{}: {:.2f}".format(class_list[one_cls_idx], one_clsprob[one_cls_idx]))
                                draw.ellipse(((center_x - 2, center_y - 2),
                                              (center_x + 2, center_y + 2)),
                                             fill='blue')

                    if  two >= objness_threshold:
                        x_start_point = dx * i
                        y_start_point = dy * j
                        x_shift = two_block[1]
                        y_shift = two_block[2]
                        center_x = int((two_block[1] * W / 7.0) + (i * W / 7.0))
                        center_y = int((two_block[2] * H / 7.0) + (j * H / 7.0))
                        w_ratio = two_block[3]
                        h_ratio = two_block[4]
                        w_ratio = w_ratio * w_ratio
                        h_ratio = h_ratio * h_ratio
                        width = int(w_ratio * W)
                        height = int(h_ratio * H)
                        xmin = center_x - (width // 2)
                        ymin = center_y - (height // 2)
                        xmax = xmin + width
                        ymax = ymin + height
                        
                        if two_clsprob[two_cls_idx] > class_threshold:
                            if (one_cls_idx == two_cls_idx) and (one_clsprob[one_cls_idx] < two_clsprob[two_cls_idx]):
                                draw.rectangle(((xmin + 2, ymin + 2), (xmax - 2, ymax - 2)), outline="blue")
                                draw.text((xmin + 5, ymin + 5), "{}: {:.2f}".format(class_list[two_cls_idx], two_clsprob[two_cls_idx]))
                                draw.ellipse(((center_x - 2, center_y - 2),
                                              (center_x + 2, center_y + 2)),
                                             fill='blue')
    
                        # LOG
                        '''
                        print("idx : [{}][{}]".format(i, j))
                        print("x shift : {}, y shift : {}".format(x_shift, y_shift))
                        print("w ratio : {}, h ratio : {}".format(w_ratio, h_ratio))
                        print("cls prob : {}".format(np.around(clsprob, decimals=2)))

                        print("xmin : {}, ymin : {}, xmax : {}, ymax : {}".format(xmin, ymin, xmax, ymax))
                        print("width : {} height : {}".format(width, height))
                        print("class list : {}".format(class_list))
                        print("\n\n\n")
                        '''
            plt.figure(figsize=(24, 18))
            plt.imshow(img)
            plt.show()
            plt.close()

        except Exception as e:
            print("ERROR")
            print("Message : {}".format(e))
