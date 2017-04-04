from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
from PIL import Image
import sys
import os
import argparse
from data import base_transform
import numpy as np
from ssd import build_ssd
import cv2

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/', type=str, help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int, help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
args = parser.parse_args()

args.trained_model = 'weights/ssd_300_VOC0712.pth'
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.eval()
transform = base_transform(net.size,(104,117,123))
colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255)]
font = cv2.FONT_HERSHEY_SIMPLEX

def predict(frame):
    #res = cv2.resize(img,(0.5*width, 0.5*height), interpolation = cv2.INTER_CUBIC)
    height,width = frame.shape[:2]
    img = Image.fromarray(frame)
    # print(img)
    x = Variable(transform(img).unsqueeze_(0))
    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width,height,width,height])
    j = 0
    for i in range(detections.size(1)):
        while detections[0,i,j,0] >= 0.3:
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), colors[i%3], 2)
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), font, 2,(255,255,255),2,cv2.LINE_AA)
            # labels.append(label_name)
            j+=1
    return frame



video_capture = cv2.VideoCapture(0)

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = predict(frame)
    # pts, labels = predict(frame)
    # color = (0, 255, 0)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # for i in range(len(pts)):
    #     cv2.rectangle(frame, (int(pts[i][0]), int(pts[i][1])), (int(pts[i][2]-pts[i][0]+1), int(pts[i][3]-pts[i][1]+1)), color, 2)
    #     cv2.putText(frame, labels[i], (pts[i][0], pts[i][1]), font, 4,(255,255,255),2,cv2.LINE_AA)
    # # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    cv2.imshow('Video', frame)
    cv2.waitKey(10)
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
