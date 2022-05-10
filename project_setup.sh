#!/bin/bash

echo "Setting up folders"

mkdir -p ./Datasets ./Datasets/coco ./Datasets/coco/images ./Datasets/coco/annotations ./Datasets/coco/images/train2017 ./Datasets/coco/images/val2017

echo ""
echo "-----------------------------"
echo " -Folders created"

echo " - Downloading annotations"

if [ -f "./Datasets/coco/annotations/lvis_v1_train.json" ]
then
    echo " - Training annotations already exist"
else
    wget -P ./Datasets/coco/annotations https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
    unzip ./Datasets/coco/annotations/lvis_v1_train.json.zip -d /Datasets/coco/annotations
    rm ./Datasets/coco/annotations/lvis_v1_train.json.zip
    echo " - Downloaded training annotations"
fi
    
if [ -f "./Datasets/coco/annotations/lvis_v1_val.json" ]
then
    echo " - Validation annotations already exist"
else
    wget -P ./Datasets/coco/annotations https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
    unzip ./Datasets/coco/annotations/lvis_v1_val.json.zip -d /Datasets/coco/annotations
    rm ./Datasets/coco/annotations/lvis_v1_val.json.zip
    echo " - Downloaded validation notations"
fi

echo ""
echo "--------------------------"
echo "File system and annotations ready to use"
echo ""
echo "--------------------------"
echo ""
echo "Please create sym link to coco image data:"
echo ""
echo "ln -s /path/to/images/train2017 /Datasets/coco/train2017"
echo "ln -s /path/to/images/val2017 /Datasets/coco/val2017" 