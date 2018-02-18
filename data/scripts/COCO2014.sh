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

end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"
