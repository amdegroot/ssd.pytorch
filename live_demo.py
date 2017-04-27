from __future__ import print_function
import sys
import os
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream

from data import BaseTransform
from data import VOC_CLASSES as labelmap
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
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                            int(pt[3])), colors[i % 3], 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), font,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame


# start video stream thread, allow buffer to fill
print("[INFO] starting threaded video stream...")
stream = WebcamVideoStream(src=0).start()  # default camera
time.sleep(1.0)

# start fps timer
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab next frame
    frame = stream.read()
    key = cv2.waitKey(1) & 0xFF

    # update FPS counter
    fps.update()
    frame = predict(frame)

    # keybindings for display
    if key == ord('p'):  # pause
        while True:
            key2 = cv2.waitKey(1) or 0xff
            cv2.imshow('frame', frame)
            if key2 == ord('p'):  # resume
                break
    cv2.imshow('frame', frame)
    if key == 27:  # exit
        break


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
stream.stop()
