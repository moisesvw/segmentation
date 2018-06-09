import os
import sys
import cv2
import numpy as np
import time
from skimage.io import imsave
from multiprocessing import Pool

ROOT_DIR = os.path.abspath('../models/Mask_RCNN')
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import utils

class SegmentationConfig(Config):
  NAME = "SegmentationConfig"
  IMAGES_PER_GPU = 2
  NUM_CLASSES = 1 + 9 # background + 9
  STEPS_PER_EPOCH = 1000 #1000

class InferenceConfig(SegmentationConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

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

class Prediction():

  def __init__(self, dataset, mask_prediction_path, cores=1):
    self.dataset = dataset
    self.mask_prediction_path = mask_prediction_path
    self.core = cores


  def find_pixels(self, mask):
      endcoded_pixels = []
      total_pixels = 0
      for i in range(mask.shape[0]):
          init = 0
          end = 0
          carry = i*mask.shape[1]
          for j in range(1, mask.shape[1]):
              if mask[i, j] > 0 and mask[i, j - 1] == 0:
                  init = j + carry

              if mask[i, j] == 0 and mask[i, j - 1] > 0:
                  end = j-init + carry
                  endcoded_pixels.append( str(init) + ' ' +  str(end) ) 
                  total_pixels += end
                  init = 0
                  end = 0

      return endcoded_pixels, total_pixels


  def process_result(self, r, image_name, image_id):
      data = []
      for i in range(len(r['scores'])):
          mask_idx = i
          confidence = r['scores'][i]
          label_id = r['class_ids'][i]
          mask = r['masks'][:, :, i]
          rois = r['rois'][i]

          data_point = {}
          old_id = self.dataset.CLASS_ID_TO_NAME[label_id]['old_id']
          mask_path = "{}{}_{}.jpg".format( self.mask_prediction_path, image_name, str(mask_idx) )
          mask_name = mask_path
          imsave(mask_name, mask)

          data_point['ImageId'] = image_name
          data_point['LabelId'] = old_id
          data_point['Confidence'] = confidence
          data_point['PixelCount'] = 0
          data_point['rois'] = rois
          data_point['EncodedPixels'] = mask_name

          data.append(data_point)

      return data

  def generate_results(self, model):
    pool = Pool(processes=8)
    test_results = []
    jobs = []
    t3 = time.time()
    for test_id in self.dataset.image_ids:
        if (test_id % 50) == 0:
            print(test_id)
        results = model.detect([self.dataset.load_image(test_id)], verbose=0)
        r = results[0]

        image_name = self.dataset.image_info[test_id]['image_name']

        p_result = pool.apply_async(self.process_result, (r, image_name, test_id) )
        jobs.append(p_result)

    for job in jobs:
        p_result = job.get(timeout=20)
        test_results = test_results + p_result
    t3_ = time.time()
    print("Time for all takes:", t3_ - t3)