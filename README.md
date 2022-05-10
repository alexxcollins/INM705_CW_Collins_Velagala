# INM705 CW: Large Vocabulary Instance Segmentation  

from the main project folder: run project_setup.sh:  

> project_setup.sh  

Either create a symlink from whereever you have your coco dataset stored:  

> ln -s /path/to/images/train2017 /Datasets/coco/train2017  
> ln -s /path/to/images/val2017 /Datasets/coco/val2017   

Or if you don't have the images installed and don't mind waiting a bit - install images from coco dataset:  

> wget -P ./Datasets/coco/images/train2017 http://images.cocodataset.org/zips/train2017.zip   
> unzip ./Datasets/coco/images/train2017/train2017.zip   
> rm ./Datasets/coco/images/train2017/train2017.zip     
> wget -P ./Datasets/coco/images/val2017 http://images.cocodataset.org/zips/val2017.zip    
> unzip ./Datasets/coco/images/val2017/val2017.zip   
> rm ./Datasets/coco/images/val2017/val2017.zip   

