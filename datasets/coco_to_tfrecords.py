# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts coco data to TFRecords file format with Example protos.

{ "segmentation":
[ # 对象的边界点（边界多边形） 
[ 224.24,297.18,# 第一个点 x,y坐标 
228.29,297.18, # 第二个点 x,y坐标 
34.91,298.29, …… …… 225.34,297.55 ]
], 
"area": 1481.3806499999994, # 区域面积 
"iscrowd": 0, # 
"image_id": 397133, # 对应的图片ID（与images中的ID对应） 
"bbox": [217.62,240.54,38.99,57.75], # 定位边框 [x,y,w,h] 
"category_id": 44, # 类别ID（与categories中的ID对应） 
"id": 82445 # 对象ID，因为每一个图像有不止一个对象，所以要对每一个对象编号（每个对象的ID是唯一的） },


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf
import cv2
import json
import numpy as np

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature


# Original dataset organisation.

#dirimg = "images/100k/val/"
#dirimg = "images/100k/train/"
#labelfile ="labels/bdd100k_labels_images_train.json"
dirimg = "images/"
#cocolabel = "/data1/dataset/coco/annotations/instances_val2017.json"
labelfile = "annotations/instances_train2017.json"
#labelfile = "annotations/instances_val2017.json"
SAMPLES_PER_FILES = 1000
SMALL_AREA=400
BDD100K_LABELS = {
    'none': (0, 'Background'),
    'motor': (1, 'Vehicle'),
    'rider': (2, 'Vehicle'),
    'car': (3, 'Vehicle'),
    'traffic sign': (9, 'Traffic'),
    'traffic light': (10, 'Traffic'),
    'bus': (6, 'Vehicle'),
    'train': (7, 'Vehicle'),
    'truck': (8, 'Vehicle'),
    'bike': (4, 'Vehicle'),
    'person': (5, 'Person'),
}
coco_categories_dict={1:'person',2:'bike',3:'car',4:'motor',6:'bus',7:'train',8:'truck'}


def _process_image(directory, labeljson):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    
    # Read the image file.
    filename = directory + labeljson['name']
    image2 = cv2.imread(filename).astype(np.uint8)
    image2 =  np.array(image2)
    image_data = image2[...,[2,1,0]]
    print(image_data.shape)
    shape=image_data.shape
    image_data = cv2.resize(image_data,(1280,720))
    image_data= image_data.tostring()
    
    imglabels=labeljson['labels']
        
    
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    print(shape)
    for label in imglabels:
        if 'bbox' in label:
            labelname = label['name']
            box = label['bbox']
            if BDD100K_LABELS[labelname][1]=='Traffic':
                print("remove traffic object")
                continue;
            
            #append
            #print("object:",labelname,BDD100K_LABELS[labelname][0])
            labels.append(int(BDD100K_LABELS[labelname][0]))
            labels_text.append(labelname.encode('ascii'))            
            bboxes.append((float(box[1]) / shape[0],
                       float(box[0]) / shape[1],
                       float(box[1]+box[3]) / shape[0],
                       float(box[0]+box[2]) / shape[1]
                       ))
            assert(float(box[1]) / shape[0]<=1)
            assert(float(box[0]) / shape[1]<=1)
            assert(float(box[1]+box[3]) / shape[0]<=1)
            assert(float(box[0]+box[2]) / shape[1]<=1)
            """print("object:",labelname,BDD100K_LABELS[labelname][0],";box:",
                       float(box[1]) / shape[0],
                       float(box[0]) / shape[1],
                       float(box[1]+box[3]) / shape[0],
                       float(box[0]+box[2]) / shape[1])"""
            difficult.append(0)
            truncated.append(0)
        
    #print("image_data",len(image_data),len(bboxes))            

        
    return image_data, [720,1280,3], bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    #image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            #'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, label, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, label)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='bdd_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames.
    print(dataset_dir)
    filename = os.path.join(dataset_dir, labelfile)
    print("filename is :",filename)
    with open(filename,'r') as load_f:
        load_dict = json.load(load_f)
    

    # Process dataset files.
    
    imgs = load_dict['images'] 
    print(len(imgs))
    imgnamedict={}
    for img in imgs:
        imgnamedict[img['id']]=img['file_name']    
    print(len(imgnamedict))
    print(len(load_dict['annotations'] ))
    print(coco_categories_dict[1])


    aimdict={}
    for annotation in load_dict['annotations']:
        if annotation['image_id'] in aimdict:
            imglabels = aimdict[annotation['image_id']]
        else:
            imglabels =[]
    
        labeldict={}
        if annotation['category_id'] in coco_categories_dict and annotation['iscrowd']==0 and annotation['area']>SMALL_AREA :
            labeldict['name']=coco_categories_dict[annotation['category_id']]
            labeldict['bbox']=annotation['bbox']
            labeldict['area']=annotation['area']
            imglabels.append(labeldict)
            aimdict[annotation['image_id']] = imglabels
    print("aimdict",len(aimdict))   

    
    
    
    
    ###################################################################
    i = 0
    fidx = 0
    imgdir = os.path.join(dataset_dir,dirimg)
    labellist=[]
    for imgid in aimdict:
        imgname = imgnamedict[imgid]
        labellist.append({'name':imgname,'labels':aimdict[imgid]})
    while i < len(aimdict):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(aimdict) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(aimdict)))
                sys.stdout.flush()

                label = labellist[i]
                
                _add_to_tfrecord(imgdir, label, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the BDD100k dataset!')
