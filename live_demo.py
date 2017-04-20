from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from sys import platform as sys_pf
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
from data import BaseTransform
from ssd import build_ssd


trained_model = 'weights/ssd_300_VOC0712.pth'
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(trained_model))
net.eval()
transform = BaseTransform(net.size, (104, 117, 123))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
font = cv2.FONT_HERSHEY_SIMPLEX


def predict(frame):
    height, width = frame.shape[:2]
    x = Variable(transform(frame).unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                          int(pt[3])), colors[i % 3], 2)
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), font,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
