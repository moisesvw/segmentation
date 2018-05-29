from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import numpy as np 
import pandas as pd
from scipy import stats
import time
import cv2
import glob

if __name__ == "__main__":
  ray.init(num_cpus=3, redirect_output=True)

  @ray.remote
  def get_masks_from_image(image_id, image):
      """
      image: numpy array containing all instances
      return numpy array of masks and array of ids
      """
      meta_data = {}
      meta_stats = {}
      stats_ = []
      ids = []
      intances_ids = np.unique(image)
      intances_ids = np.delete(intances_ids, np.where(intances_ids == 255))
      masks_shape = (image.shape[0], image.shape[1], intances_ids.shape[0])
      masks = np.zeros(masks_shape, dtype=np.bool)
      meta_data['image'] = image_id

      for instance in intances_ids:
          class_id = int(instance/1000)
          s_class_id = str(class_id)
          meta_data[s_class_id] = 0
          meta_stats[s_class_id] =[]

      for i in range(intances_ids.shape[0]):
          class_id = int(intances_ids[i]/1000)
          ids.append(class_id)
          meta_data[str(class_id)] += 1

          masks[:, :, i] = (image == intances_ids[i])
          meta_stats[str(class_id)].append(masks[:, :, i].sum())

      for key in meta_stats.keys():
          obs = {}
          mode = stats.mode(meta_stats[key])
          desc = stats.describe(meta_stats[key])
          obs['image'] = image_id
          obs['id'] = key
          obs['mode'] = mode.mode[0]
          obs['min'] = desc.minmax[0]
          obs['max'] = desc.minmax[1]
          obs['variance'] = round(desc.variance, 4)
          obs['skewness'] = round(desc.skewness, 4)
          obs['mean'] = round(desc.mean, 4)
          obs['kurtosis'] = round(desc.kurtosis, 4)
          stats_.append(obs)
          

      return masks, ids, meta_data, stats_

  train_labels = glob.glob('../data/train_label/*.png')

  data_ = []
  stats_obs = []
  start_time = time.time()
  rids = []
  for label in train_labels[:5]:
      im = cv2.imread(label, cv2.IMREAD_UNCHANGED)
      start_time_in = time.time() 
      rid = get_masks_from_image.remote(label, im)
      end_time_in = time.time() 
      print("This sequence holds element:", end_time_in-start_time_in )
      rids.append(rid)


  results = ray.get(rids)
  print(results)

  end_time = time.time()
  print("This sequence holds:", end_time-start_time )