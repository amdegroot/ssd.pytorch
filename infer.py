#do the infer

import torch
import cv2
from ssd import build_ssd

num_classes = 201
image = cv2.imread("data/1.jpg")
weights = "weights/ssd300_COCO_10000.pth"

#cv2.imshow("fafda", image)
#cv2.waitKey()

#def infer()
def get_features_hook(self, input, output):
    print("hooks ", output.data.cpu().numpy().shape)

if __name__ == '__main__':
    net = build_ssd('test', 300, num_classes)
    image = cv2.resize(image, (300, 300))
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    #load weights to the net
    #print(net.load_state_dict(torch.load(weights)))
    net.load_state_dict(torch.load(weights))
    #handle = net.ssd[0].register_forward_hook(get_features_hook)
    _ = net(image)
#    handle.remove()
#get the specific layer value
#    print(net)
