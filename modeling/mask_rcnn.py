import os
import sys
import random
import cv2
import math
import time
import numpy as np
import re
import glob

ROOT_DIR = os.path.abspath('../models/Mask_RCNN')
#import mask rcnn
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class SegmentationConfig(Config):
  NAME = "SegmentationConfig"
  IMAGES_PER_GPU = 2
  NUM_CLASSES = 1 + 9 # background + 9

class SegmentationDataset(utils.Dataset):
  ALL_CLASSES = [
              ('bicycle', 35),
              ('bicycle_group', 163),
              ('billboard', 86),
              ('bridge', 98),
              ('building', 97),
              ('bus', 39),
              ('bus_group', 167),
              ('car', 33),
              ('car_groups', 161),
              ('dustbin', 85),
              ('fence', 67),
              ('motorbicycle', 34),
              ('motorbicycle_group', 162),
              ('others', 0),
              ('overpass', 100),
              ('person', 36),
              ('person_group', 164),
              ('pole', 82),
              ('rider', 37),
              ('rider_group', 165),
              ('road', 49),
              ('road_pile', 66),
              ('rover', 1),
              ('siderwalk', 50),
              ('sky', 17),
              ('traffic_cone', 65),
              ('traffic_light', 81),
              ('traffic_sign', 83),
              ('tricycle', 40),
              ('tricycle_group', 168),
              ('truck', 38),
              ('truck_group', 166),
              ('tunnel', 99),
              ('vegatation', 113),
              ('wall', 84)
            ]

  MAIN_CLASSES = [33, 35, 39, 40, 36, 65, 34, 38, 37]

  CLASSES = [ c for c in ALL_CLASSES if c[1] in [33, 35, 39, 40, 36, 65, 34, 38, 37] ]
  CLASS_NAME_TO_ID = {}
  CLASS_ID_TO_NAME = {}
  CLASS_OLD_ID_TO_ID = {}
  for i in range( len(CLASSES) ):
      c_name = CLASSES[i][0]
      old_id = CLASSES[i][1]
      new_id = i + 1

      CLASS_NAME_TO_ID[c_name] = { 'id': new_id,   'old_id': old_id }
      CLASS_ID_TO_NAME[new_id] = { 'name': c_name, 'old_id': old_id }
      CLASS_OLD_ID_TO_ID[old_id] = { 'name': c_name, 'id': new_id }

  def load_cvpr_images(self, images_paths):
    # Add classes
    for class_name in self.CLASS_NAME_TO_ID.keys():
      self.add_class("wad", self.CLASS_NAME_TO_ID[class_name]['id'], class_name)

    for i in range(len(images_paths)):
      path = images_paths[i]
      image_name = path.split('/')[-1].split('.')[0]
      self.add_image("wad", image_id=i, path=path, image_name=image_name)

  def load_mask(self, image_id):
    image_name = self.image_info[image_id]['image_name']
    path = "../data/train_label/"+ image_name +"_instanceIds.png"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    ids = []

    all_instances = np.unique(image)
    intances_ids = [ int(x/1000) for x in all_instances ]
    intances_ids = np.array([x for x in intances_ids if x in self.MAIN_CLASSES ])

    masks_shape = (image.shape[0], image.shape[1], intances_ids.shape[0])
    masks = np.zeros(masks_shape, dtype=np.bool)

    i_ = 0
    for i in range(all_instances.shape[0]):
      class_id = int(all_instances[i]/1000)
      if self.CLASS_OLD_ID_TO_ID.get(class_id):
        ids.append(self.CLASS_OLD_ID_TO_ID[class_id]['id'])
        masks[:, :, i_] = (image == all_instances[i])
        i_+=1

    ids = np.array(ids, dtype=np.int32)
    return masks, ids


config = SegmentationConfig()
config.display()
dataset_train = SegmentationDataset()
dataset_train.load_cvpr_images(glob.glob('../data/train_color/*.jpg')[:500])
dataset_train.prepare()

dataset_val = SegmentationDataset()
dataset_val.load_cvpr_images(glob.glob('../data/train_color/*.jpg')[:100])
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')