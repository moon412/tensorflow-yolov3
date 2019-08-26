## Introduction
This repo is forked from [YunYang1994](https://github.com/YunYang1994/tensorflow-yolov3.git). The publication for YOLOV3 is [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf). Some details in the paper may be hard to understand without looking at the actual code and implementation. Because YOLO was originally implemented in DarkNet by the author. This repo is one of many TensorFlow implementations. I haven't compared this repo's implementations with DarkNet implementations. But I think this is a good repo to start with for someone like me who has some background in machine learning but is new to object detection. So a few notebooks are added to decompose the code and understand how YOLOV3 works from a implementation perspective.

## Part 1. Quick start
1. Clone this repo
```bashrc
$ git clone https://github.com/moon412/tensorflow-yolov3.git
```

2. Load the pre-trained TF checkpoint(`yolov3_coco.ckpt`) and export a .pb file. The checkpoint is provided from the forked repo not from the YOLO author though. 
```bashrc
$ cd checkpoint
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py
$ python freeze_graph.py
```
3. Then you will get the `.pb` file in the root path.,  and run the demo script
```bashrc
$ python image_demo.py
$ python video_demo.py # if use camera, set video_path = 0
```
4. Load the checkpoint file and export the SaveModel object to the `savemodel` folder for TensorFlow serving
```bashrc
$ python save_model.py
```

5. Copy the `yolov3` folder to `tmp` and serve the model as a RESTful API with [TF serving](https://www.tensorflow.org/tfx/serving/setup)
```
$ docker run -p 8501:8501 --mount type=bind,source=/tmp/yolov3/,target=/models/yolov3 -e MODEL_NAME=yolov3 -t tensorflow/serving &
```
Look at [test_tf_serving.ipynb](https://github.com/moon412/tensorflow-yolov3/blob/master/test_tf_serving_api.ipynb) for details to serve and test the model. 
To serve the model, I added one variable `pred_multi_scale` to [class YOLOV3](https://github.com/moon412/tensorflow-yolov3/blob/master/core/yolov3.py#L60). This variable concatenates the tensors from the three scales (`pred_sbbox`, `pred_mbbox` and `pred_lbbox`) into one single tensor. Because, as far as I know, TF serving only outputs one single tensor. This is quite hacky (you may notice the dimensions are hard-coded for 606 * 608 * 3 images) but it works for now. 

## Part 2. Training
### 2.1 Two files are required as follows:

- [`dataset.txt`](https://raw.githubusercontent.com/moon412/tensorflow-yolov3/master/data/dataset/voc_train.txt): 

```
In voc_train.txt
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# x_min, y_min etc. corresponds to the data in XML files
```

- [`class.names`](https://github.com/moon412/tensorflow-yolov3/blob/master/data/classes/coco.names):

```
person
bicycle
car
...
toothbrush
``` 

### 2.2 Download VOC dataset and prepare the above files
Download VOC PASCAL trainval and test data
```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory and rename them, which should have the following basic structure.

```bashrc

VOC           # path:  /home/yang/test/VOC/
├── test
|    └──VOCdevkit
|       └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
             └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
                     └──VOC2012 (from VOCtrainval_11-May-2012.tar)
```
Use the following script to write the paths to the training images and the corresponding annotations to dataset.txt as
```bashrc
                     
$ python scripts/voc_annotation.py --data_path /home/yang/test/VOC
```
Then edit your `./core/config.py` to make some necessary configurations

```bashrc
__C.YOLO.CLASSES                = "./data/classes/voc.names"
__C.TRAIN.ANNOT_PATH            = "./data/dataset/voc_train.txt"
__C.TEST.ANNOT_PATH             = "./data/dataset/voc_test.txt"
```
One thing I haven't figured out for training is how to prepare the training set for multi-scaling. I thought even with 3 scales (3 grids), only one anchor box in one grid cell is responsible for one object. But in [dataset.ipynb](https://github.com/moon412/tensorflow-yolov3/blob/master/dataset.ipynb), for the image with two persons and a horse (cell 53-56), two scales have object confidence scores equal to 1 for the same objects.

### 2.3 Training from scratch:

```bashrc
$ python train.py
$ tensorboard --logdir ./data
```

### 2.4 Evaluation
```
$ python evaluate.py
$ cd mAP
$ python main.py -na
```

### 2.5 Train with other datasets
Download COCO trainval  and test data
```
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/zips/test2017.zip
$ wget http://images.cocodataset.org/annotations/image_info_test2017.zip 
```



