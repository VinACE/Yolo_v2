# -*- coding:utf-8 -*-

import argparse

from train import train
from test import test

parser = argparse.ArgumentParser(description='YOLO v2.')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--dataset', type=str, help='dataset to train on, voc', default='voc')
parser.add_argument('--data_path', type=str, help='path to the data', default='/home/madhevan/mrcnn/IDRBTcheque/train/', required=True)
parser.add_argument('--val_data_path', type=str, help='path to the data', default='/home/madhevan/mrcnn/IDRBTcheque/test/', required=False)
parser.add_argument('--val_datalist_path', type=str, help='path to the data', default='./', required=False)
parser.add_argument('--datalist_path', type=str, help='path to use dataset list to train', required=False, default="./")
parser.add_argument('--class_path', type=str, help='path to the filenames text file',default='/home/madhevan/mrcnn/IDRBTcheque/train/classes.txt', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=416)
parser.add_argument('--input_width', type=int, help='input width', default=416)
parser.add_argument('--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=101)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='./')

# flag
parser.add_argument('--use_augmentation', type=lambda x: (str(x).lower() == 'true'), help='Image Augmentation', default=False)
parser.add_argument('--use_gtcheck', type=lambda x: (str(x).lower() == 'true'), help='Ground Truth check flag', default=False)
parser.add_argument('--use_githash', type=lambda x: (str(x).lower() == 'true'), help='use githash to checkpoint', default=False)
parser.add_argument('--use_visdom', type=lambda x: (str(x).lower() == 'true'), help='use githash to checkpoint', default=False)

# develop
parser.add_argument('--num_class', type=int, help='number of class', default='5', required=True)
args = parser.parse_args()


def main():
    params = {
        "mode": args.mode,
        "dataset": args.dataset,
        "data_path": args.data_path,
        "val_data_path":args.val_data_path,
        "val_datalist_path":args.val_datalist_path,
        "datalist_path": args.datalist_path,
        "class_path": args.class_path,
        "input_height": args.input_height,
        "input_width": args.input_width,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.learning_rate,
        "checkpoint_path": args.checkpoint_path,

        "use_augmentation": args.use_augmentation,

        "num_class": args.num_class,
        "use_gtcheck": args.use_gtcheck,
        "use_githash": args.use_githash,
        "use_visdom": args.use_visdom
    }

    if params["mode"] == "train":
        train(params)
    elif params["mode"] == "test":
        test(params)


if __name__ == '__main__':
    main()
