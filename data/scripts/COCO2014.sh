#!/bin/bash

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ~/data/ ..."
    mkdir -p ~/data
    cd ~/data/
    mkdir -p ./coco
    cd ./coco
    mkdir -p ./images
    mkdir -p ./annotations
  else
    # check if specified dir is valid
    if [ ! -d $1 ]; then
        echo $1 " is not a valid directory"
        exit 0
    fi
    echo "navigating to " $1 " ..."
    cd $1
fi

if [ ! -d images ]
  then
    mkdir -p ./images
fi

# Download the image data.
cd ./images
echo "Downloading MSCOCO train images ..."
curl -LO http://images.cocodataset.org/zips/train2014.zip
echo "Downloading MSCOCO val images ..."
curl -LO http://images.cocodataset.org/zips/val2014.zip

cd ../
if [ ! -d annotations]
  then
    mkdir -p ./annotations
fi

# Download the annotation data.
cd ./annotations
echo "Downloading MSCOCO train/val annotations ..."
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2014.zip
echo "Finished downloading. Now extracting ..."

# Unzip data
echo "Extracting train images ..."
unzip ../images/train2014.zip -d ../images
echo "Extracting val images ..."
unzip ../images/val2014.zip -d ../images
echo "Extracting annotations ..."
unzip ./annotations_trainval2014.zip

echo "Removing zip files ..."
rm ../images/train2014.zip
rm ../images/val2014.zip
rm ./annotations_trainval2014.zip

echo "Creating trainval35k dataset..."

# Download annotations json
echo "Downloading trainval35k annotations from S3"
wget https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip
unzip instances_valminusminival2014.json.zip
mv instances_valminusminival2014.json instances_trainval35k.json

# combine train and val 
echo "Combining train and val images"
mkdir ../images/trainval35k
cd ../images/train2014
find -maxdepth 1 -name '*.jpg' -exec cp -t ../trainval35k {} + # dir too large for cp
cd ../val2014
find -maxdepth 1 -name '*.jpg' -exec cp -t ../trainval35k {} +


end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"
