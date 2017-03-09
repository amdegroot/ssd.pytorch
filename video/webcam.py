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


stream=urllib.request.urlopen("XXXXX")
bytes=''

def predict(frame):

    # Capture frame-by-frame

    res = predict(net,image)
    color = (0, 255, 0)
    cv2.rectangle(image, (res[0], res[1]), (res[2]-res[0]+1, res[3]-res[1]+1), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, res[4], (res[0], res[1]), font, 4,(255,255,255),2,cv2.LINE_AA)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #image = Image.open(args.image_file).convert('RGB')
    height, width = frame.shape[:2]

    print(height)
    print(width)
    #res = cv2.resize(img,(0.5*width, 0.5*height), interpolation = cv2.INTER_CUBIC)
    im = Image.fromarray(frame)
    t = test_transform(300,(104,117,123))
    x = t(im)
    x = Variable(x) # wrap tensor in Variable
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

    return (x1,y1,x2,y2, label)


    app = Flask(__name__)

    @app.route('/')
    def index_final():
    	return render_template('index_final.html')

    def gen():
    	net = build_ssd('test', 300, 21)
    	net.load_weights('weights/ssd_300_voc07.pkl')
    	while True:
            bytes+=stream.read(1024)
            a = bytes.find('\xff\xd8')
            b = bytes.find('\xff\xd9')
            if a!=-1 and b!=-1:
                count+=1
                jpg = bytes[a:b+2]
                bytes= bytes[b+2:]
                i  = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
                if count % 20 == 0:
                    res = predict(net,i)
                    color = (0, 255, 0)
                    cv2.rectangle(image, (res[0], res[1]), (res[2]-res[0]+1, res[3]-res[1]+1), color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, res[4], (res[0], res[1]), font, 4,(255,255,255),2,cv2.LINE_AA)
                    ret, jpeg = cv2.imencode('.jpg',image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r1\n')



    @app.route('/video_final')
    def video_final():
    	return Response(gen(),
    		mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	# net = build_ssd(300,'test',21)
	# net.load_weights()
	app.run(host='localhost', port = 8888, debug=True, threaded=True)
