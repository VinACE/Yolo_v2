import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import darknet

def main():
    imageNet_label = [line.strip() for line in open("/home/madhevan/mrcnn/IDRBTcheque/demo/classes.txt", 'r')]
    import pdb;pdb.set_trace()
    dataset = dset.ImageFolder(root="/home/madhevan/mrcnn/IDRBTcheque/demo",
                           transform=transforms.Compose([
                               transforms.Resize((448, 448)),
                               transforms.ToTensor()
                           ]))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # darknet19 = darknet.Darknet19()
    # checkpoint = torch.load("/home/madhevan/Yolo_v2/ckpt_noHash_ep00010_loss103.5820_lr0.0001.pth.tar")
    # try:
    #     checkpoint.eval()
    # except AttributeError as error:
    #     print(error)

    # # ### 'dict' object has no attribute 'eval'
    # darknet19.load_state_dict(checkpoint['state_dict'])
    # # model.load_state_dict(torch.load(’/path/to/model.pth.tar’))
    # # https://discuss.pytorch.org/t/fine-tuning-resnet-dataparallel-object-has-no-attribute-fc/14842/3
    # darknet19.eval()
    # # darknet19 =  checkpoint

    darknet19 = darknet.Darknet19(pretrained=True)
    darknet19.eval()
    for data, _ in dataloader:
        output = darknet19.forward(data)
        #output = checkpoint.forward(data)
        answer = int(torch.argmax(output))
        print("Class: {}({})".format(imageNet_label[answer],answer))
        plt.imshow(np.array(np.transpose(data[0], (1, 2, 0))))
        plt.show()

if __name__ == "__main__":
    main()
