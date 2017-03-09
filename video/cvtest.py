
import os
import cv2
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
import torch.utils.data as data
from PIL import Image
import sys
import os
from data import AnnotationTransform, VOCDetection, test_transform
from ssd import build_ssd
from timeit import default_timer as timer
from data import VOC_CLASSES as labelmap
import numpy as np
import urllib.request

net = build_ssd('test', 300, 21)
net.load_weights('weights/ssd_300_voc07.pkl')


def predict(frame):
    #res = cv2.resize(img,(0.5*width, 0.5*height), interpolation = cv2.INTER_CUBIC)
    height = frame.shape[0]
    width = frame.shape[1]
    im = Image.fromarray(frame)

    t = test_transform(300,(104,117,123))
    x = t(im)
    x = Variable(x.unsqueeze(0)) # wrap tensor in Variable
    y = net(x)      # forward pass
    detections = y.data.cpu().numpy()
    # Parse the outputs.
    det_label = detections[0,:,1]
    det_conf = detections[0,:,2]
    det_xmin = detections[0,:,3]
    det_ymin = detections[0,:,4]
    det_xmax = detections[0,:,5]
    det_ymax = detections[0,:,6]

    label = labelmap[int(det_label[0])-1]
    score = det_conf[0]
    x1 = int(det_xmin[0]*height)
    y1 = int(det_ymin[0]*width)
    x2 = int(det_xmax[0]*height)
    y2 = int(det_ymax[0]*width)
    return (x1,y1,x2,y2,label)



video_capture = cv2.VideoCapture(0)
# anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    res = predict(frame)
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (res[0], res[1]), (res[2]-res[0]+1, res[3]-res[1]+1), color, 2)
    cv2.putText(frame, res[4], (res[0], res[1]), font, 4,(255,255,255),2,cv2.LINE_AA)
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
