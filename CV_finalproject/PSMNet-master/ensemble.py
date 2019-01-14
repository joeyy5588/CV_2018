from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from utils import preprocess 
from models import *
from util import writePFM
from dataloader import listflowfile as lt
import modifylistflowfile as mlt
from dataloader import SecenFlowLoader as DA
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = mlt.dataloader(args.datapath)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, True), 
         batch_size= 1, shuffle= False, num_workers= 4, drop_last=False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

epoch_list = [990, 986, 988, 922, 991, 994, 993, 987]
epoch_list = epoch_list[:4]
model_list = []
for i in range(len(epoch_list)):
    if args.model == 'stackhourglass':
        model = stackhourglass(args.maxdisp)
    elif args.model == 'basic':
        model = basic(args.maxdisp)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()
    state_dict = torch.load('trained/checkpoint_' + str(epoch_list[i]) + '.tar')
    model.load_state_dict(state_dict['state_dict'])
    model_list.append(model)
    #print(model_list)

#print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR, model):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()
        return pred_disp
def cal_avgerr(GT, disp):
    return np.sum(np.multiply(np.abs(GT - disp), GT)) / np.sum(GT)
def main():
    total_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        pred_disp = 0
        min_loss = 10
        disp_GT = disp_L.cpu().numpy().squeeze(0)
        for i in range(len(epoch_list)):
            model = model_list[i]
            temp_disp = test(imgL,imgR, model)
            loss = cal_avgerr(disp_GT, temp_disp)
            if loss < min_loss:
                pred_disp = temp_disp
                min_loss = loss
        total_loss += min_loss
        print(min_loss)
        writePFM('output/TL' + str(batch_idx) + '.pfm', pred_disp)
    print('total loss: ', total_loss/10)

if __name__ == '__main__':
   main()