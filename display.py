import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from ssd import build_ssd
import boto3
import tempfile


def get_img_targ_from_s3(img_id, s3_bucket='geniussports-computer-vision-data',
                         s3_path='internal-experiments/basketball/bhjc/20180123/',
                         file_name_prfx='left_scene2_rot180_'):

    im_path = s3_path + 'images/left_cam/' + file_name_prfx + img_id + '.png'
    print('loading:', im_path)

    s3 = boto3.resource('s3', region_name='us-west-2')
    bucket = s3.Bucket(s3_bucket)
    im_obj = bucket.Object(im_path)
    tmp = tempfile.NamedTemporaryFile()

    # dowload to temp file and read in
    with open(tmp.name, 'wb') as f:
        im_obj.download_fileobj(f)
    img = cv2.imread(tmp.name)

    return img


net = build_ssd('test', 300, 3, square_boxes=False)    # initialize SSD 21 classes (num classes + 1)
weight_file = 'weights/ssd1166_bhjctrained_iter5000_smallLR.pth'
net.load_weights(weight_file)

from matplotlib import pyplot as plt

im00 = '00{}'

for id in range(650, 755):
    img_id = im00.format(id)

# img_id = '00670'
#     path_to_bball_im = '/Users/keith.landry/data/internal-experiments/basketball/bhjc/20180123/images/left_cam/'
#     bball_file = 'left_scene2_rot180_{}.png'.format(img_id)  # 442, 531, 855, 132
#     image = cv2.imread(path_to_bball_im + bball_file, cv2.IMREAD_COLOR)

    image = get_img_targ_from_s3(img_id)
    print(image.shape)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (1166, 1166)).astype(np.float32)
    # x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104, 117, 123)  #VOC means
    # x -= (103, 100, 94)   #BHJC means

    x = x[:, :, ::-1].copy()
    x = x.astype(np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x.shape

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)  # passes image through pretrained network

    # from data import VOC_CLASSES as labels
    from data.bhjc20180123_bball.bhjc import CLASSES as labels

    top_k = 10

    plt.figure(figsize=(10, 10))
    colors = ['green', 'purple'] #plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(1, detections.size(1)):  # i should start at 1 because 0 is background

        top_det_scores = detections[0, i, :2, 0]
        top_det_locats = detections[0, i, :2, 1:]

        for score, loc in zip(top_det_scores, top_det_locats):
            label_name = labels[i - 1]
            print(label_name, score)
            if score >= .194:
                display_txt = '%s: %.2f' % (label_name, score)
                pt = (loc * scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                color = colors[i-1]
                # if i == 2:
                #     color = 'purple'
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False,
                                                    edgecolor=color, linewidth=1, alpha=.8))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.2})

    outfile = '/home/ec2-user/computer_vision/bball_detection/ssd.pytorch/data/output_imgs/leftcam_detect_{}.png'.format(img_id)
    # outfile = '/Users/keith.landry/code/ssd.pytorch/data/output_imgs/leftcam_detect_{}.png'.format(img_id)
    plt.savefig(outfile)
