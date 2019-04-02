# calculate dataset mean of three channel
import os
import numpy as np
import cv2

rootpath='/media/mk/本地磁盘/Datasets/UAV/VisDrone2018/VisDrone2018-VID-train/sequences/'

R=0
G=0
B=0
N=0
for subfoldername in os.listdir(rootpath):
    for imgname in os.listdir(os.path.join(rootpath, subfoldername)):
        print('N: '+str(N))
        img=cv2.imread(os.path.join(rootpath, subfoldername, imgname))
        xr=(np.sum(img[:,:,0]))/(img.shape[0]*img.shape[1])
        R=R*(N/(N+1))+xr/(N+1)

        xg=np.sum(img[:,:,1])/(img.shape[0]*img.shape[1])
        G=G*(N/(N+1))+xg/(N+1)

        xb=np.sum(img[:,:,2])/(img.shape[0]*img.shape[1])
        B=B*(N/(N+1))+xb/(N+1)

        N+=1

print(R)
print(G)
print(B)

