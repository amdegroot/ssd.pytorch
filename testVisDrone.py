from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import shutil

torch.set_default_tensor_type('torch.cuda.FloatTensor') # 一定要注意设置cuda，不然是torch.FloatTensor

def eval(trained_model, figsize=300, figmean=(119, 122, 116)): # resize的大小，均值
    valpath='/media/mk/本地磁盘/Datasets/UAV/VisDrone2018/VisDrone2018-VID-val/'
    seqpath=os.path.join(valpath, 'sequences/')
    annopath=os.path.join(valpath, 'annotations/')
    resultpath=trained_model.split('/')[1]+'_results'

    # 构建网络
    num_classes = len(DRONE_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(trained_model))
    net.eval()
    net = net.cuda()
    cudnn.benchmark = True
    print('Finished loading model!')

    annotransform=DroneAnnotationTransform()    # box已经进行标准化（小数），不需要再使用imgtransform的变换
    imgtransform=BaseTransform(figsize, figmean)

    for clip in os.listdir(seqpath):
        print('Clip: '+clip)
        annoname=os.path.join(annopath, clip)
        resname=os.path.join(resultpath, 'res_'+clip+'.txt')

        for frame in sorted(os.listdir(os.path.join(seqpath, clip))):

            # get Groundtruth
            imgname=os.path.join(seqpath, clip, frame)
            frameidx=int(frame.split('.')[0])
            # print('idx: '+str(frameidx))
            img=cv2.imread(imgname)
            height, width, channel=img.shape
            boxes=annotransform(annoname, frameidx, width, height)

            # get Pred
            x=torch.from_numpy(imgtransform(img)[0]).permute(2, 0, 1).cuda()    # 创建时就是cuda tensor
            x=torch.Tensor(x.unsqueeze(0))  # 在0维增加一个维度， 3x300x300 -> 1x3x300x300
            x=x.cuda()

            y=net(x)
            detections=y.data
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                img.shape[1], img.shape[0]])

            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.6:
                    score = detections[0, i, j, 0]
                    label = str(i)
                    pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                    bbox = (pt[0], pt[1], pt[2]-pt[0], pt[3]-pt[1])
                    with open(resname, mode='a+') as f:
                        f.write(str(frameidx) + ',-1,' + ','.join(str(c) for c in bbox) + ',' + 
                                str(round(float(score.data), 4)) + ',' + label + ',-1,-1\n')
                    # print(str(frameidx) + ',-1,' + ','.join(str(c) for c in bbox) + ',' + 
                    #             str(round(float(score.data), 4)) + ',' + label + ',-1,-1')
                    print(' '+label+': '+str(round(float(score.data), 4)))
                    j += 1


if __name__ == '__main__':
    trained_model='weights/VisDrone2018_300.pth'

    if os.path.exists(trained_model.split('/')[1]+'_results'):
        shutil.rmtree(trained_model.split('/')[1]+'_results')
    os.mkdir(trained_model.split('/')[1]+'_results')

    eval(trained_model)


# def test_net(save_folder, net, cuda, testset, transform, thresh):
#     # dump predictions and assoc. ground truth to text file for now
#     filename = save_folder+'test1.txt'
#     num_images = len(testset)
#     for i in range(num_images):
#         print('Testing image {:d}/{:d}....'.format(i+1, num_images))
#         img = testset.pull_image(i)
#         img_id, annotation = testset.pull_anno(i)   # id， bbox， 类别
#         x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
#         x = Variable(x.unsqueeze(0))

#         with open(filename, mode='a') as f:
#             f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
#             for box in annotation:
#                 f.write('label: '+' || '.join(str(b) for b in box)+'\n')
#         if cuda:
#             x = x.cuda()

#         y = net(x)      # forward pass
#         detections = y.data

#         print('network output: '+str(y.data))

#         # scale each detection back up to the image
#         scale = torch.Tensor([img.shape[1], img.shape[0],
#                              img.shape[1], img.shape[0]])
#         pred_num = 0
#         for i in range(detections.size(1)):
#             j = 0
#             while detections[0, i, j, 0] >= 0.6:
#                 if pred_num == 0:
#                     with open(filename, mode='a') as f:
#                         f.write('PREDICTIONS: '+'\n')
#                 score = detections[0, i, j, 0]
#                 label_name = labelmap[i-1]
#                 pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
#                 coords = (pt[0], pt[1], pt[2], pt[3])
#                 pred_num += 1
#                 with open(filename, mode='a') as f:
#                     f.write(str(pred_num)+' label: '+label_name+' score: ' +
#                             str(score.data) + ' '+' || '.join(str(c) for c in coords) + '\n')
#                 j += 1

