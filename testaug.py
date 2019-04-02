# Test augmentation for single object

import cv2
import torch
import os
import numpy as np
from utils.augmentations import SSDAugmentation

rootpath='/media/mk/本地磁盘/Datasets/UAV/VisDrone2018/VisDrone2018-VID-train/sequences/uav0000013_00000_v/'

DRONE_CLASSES = (  #  1+11=12
    'pedestrian', 'person', 'bicycle', 'car',
    'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
    'motor', 'others')
    
anno=[]

with open('/media/mk/本地磁盘/Datasets/UAV/VisDrone2018/VisDrone2018-VID-train/annotations/uav0000013_00000_v.txt', 'r') as f:
    content=f.readlines()
    for line in content:
        line = line.strip().split(',')  # 去换行符，逗号分割
        anno.append([int(line[0]), float(line[2]), float(line[3]), float(line[2])+float(line[4]), float(line[3])+float(line[5]),int(line[7])])

def getBoxLab(idx, width, height):
    box=[]
    lab=[]
    for item in anno:
        # print(str(item[0])+' '+str(idx))
        if item[0]==int(idx):   # 可能有多个物体
            box.append([item[1]/width, item[2]/height, item[3]/width, item[4]/height])
            
            lab.append(item[5])

    return np.array(box), np.array(lab) # TypeError: list indices must be integers or slices, not tuple 原因是没有将list转为np.array

aug=SSDAugmentation()
cnt=0
for filename in os.listdir(rootpath):
    if cnt<10:
        cnt+=1
    else:
        break

    frameidx=int(filename.split('.')[0])
    # print(frameidx)
    img=cv2.imread(os.path.join(rootpath, filename), cv2.IMREAD_COLOR)
    width=img.shape[1]
    height=img.shape[0] # 转为相对表示
    box, lab=getBoxLab(frameidx, width, height)
    # print(box[:][0])    # list的索引[:][0]，多维index； array索引，逗号tuple [:,1]
                        # TypeError: list indices must be integers or slices, not tuple 原因是没有将list转为np.array

    # print(np.array(box).shape)

    # aimg=img
    # aimg=np.array(aimg)
    # for it in box:
    #     print(it)
    #     img=cv2.rectangle(aimg, (int(it[0].item()*aimg.shape[1]), int(it[1].item()*aimg.shape[0])), (int(it[2].item()*aimg.shape[1]), int(it[3].item()*aimg.shape[0])), (0,0,255), 4)
    #     # 只能用python的数据类型而不能用numpy的数据类型，只能使用int，而不能使用float
    # cv2.imshow('aaa', aimg)
    # cv2.waitKey()   # 加上waitkey才显示图片    

    # cv2.imshow('origin', img)
    # cv2.waitKey()   # 加上waitkey才显示图片

    newimg, newbox, newlab=aug(img, box, lab)   # newimg (300,300,4), newbox(27,4)

    imgheight, imgwidth, imgchannel=newimg.shape
    origin=np.copy(newimg)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # if abs(np.max(origin)-int(np.max(origin)))>=1e-7: # 判断是真小数
    #     print('OK')
    origin=origin/origin.max()  # cv2画图： 浮点数必须0-1.0， 整数0-255
    for idx, it in enumerate(newbox):
        if float(it[0].item()) <= 1.0:  
            origin=cv2.rectangle(origin, (int(it[0].item()*imgwidth), int(it[1].item()*imgheight)), (int(it[2].item()*imgwidth), int(it[3].item()*imgheight)), (255,255,0), 4)
            origin=cv2.putText(origin, DRONE_CLASSES[int(newlab[idx])], (int(it[0].item()*imgwidth), int(it[1].item()*imgheight)-2), font, 1, (0,0,255), 1)
        else:
            origin=cv2.rectangle(origin, (int(it[0].item()), int(it[1].item())), (int(it[2].item()), int(it[3].item())), (255,255,0), 4)
            origin=cv2.putText(origin, DRONE_CLASSES[int(newlab[idx])], (int(it[0].item()), int(it[1].item())-2), font, 1, (0,0,255), 1)
    
    print(origin[0][0])
    cv2.imshow('ttt', origin)
    cv2.waitKey()   # 加上waitkey才显示图片  

