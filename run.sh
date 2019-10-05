#!/bin/bash

rm -r logs/
rm trainlogs.txt
mkdir logs/

python train.py --dataset VisDrone2018 &amp;

echo "start training"

tensorboard --logdir='./logs'
google-chrome http://mk-alphago:6006/#scalars
