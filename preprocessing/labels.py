from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np 
import pandas as pd
from scipy import stats
import time
import cv2
import glob
import ray
"""
This is going to be the job thar will be
share accross workers.
This contain other functions.
"""

if __name__ == '__main__':
  """
  This is the driver program
  """
  ray.init()

  @ray.remote
  def get_masks_from_image(image_id):
      """
      image: numpy array containing all instances
      return numpy array of masks and array of ids
      """
      t1 = time.time()
      image = cv2.imread(image_id, cv2.IMREAD_UNCHANGED)
      t1_ = time.time()
      print("Reading image takes:", t1_- t1 )
      meta_data = {}
      meta_stats = {}
      stats_ = []
      ids = []

      t2 = time.time()
      intances_ids = np.unique(image)
      t2_ = time.time()
      print("Numpy unique takes:", t2_- t2 )

      t3 = time.time()
      intances_ids = np.delete(intances_ids, np.where(intances_ids == 255))
      t3_ = time.time()
      print("Delete index takes:", t3_- t3 )


      masks_shape = (image.shape[0], image.shape[1], intances_ids.shape[0])
      masks = np.zeros(masks_shape, dtype=np.bool)
      meta_data['image'] = image_id

      for instance in intances_ids:
          class_id = int(instance/1000)
          s_class_id = str(class_id)
          meta_data[s_class_id] = 0
          meta_stats[s_class_id] =[]

      t4 = time.time()
      for i in range(intances_ids.shape[0]):
          class_id = int(intances_ids[i]/1000)
          ids.append(class_id)
          meta_data[str(class_id)] += 1

          masks[:, :, i] = (image == intances_ids[i])
          meta_stats[str(class_id)].append(masks[:, :, i].sum())
      t4_ = time.time()
      print("Creating mask takes:", t4_- t4 )

      t5 = time.time()
      for key in meta_stats.keys():
          obs = {}
          mode = stats.mode(meta_stats[key])
          # desc = stats.describe(meta_stats[key])
          # obs['image'] = image_id
          # obs['id'] = key
          obs['mode'] = mode.mode[0]
          # obs['min'] = desc.minmax[0]
          # obs['max'] = desc.minmax[1]
          # obs['variance'] = round(desc.variance, 4)
          # obs['skewness'] = round(desc.skewness, 4)
          # obs['mean'] = round(desc.mean, 4)
          # obs['kurtosis'] = round(desc.kurtosis, 4)
          stats_.append(obs)
      t5_ = time.time()
      print("Adding Stats takes:", t5_- t5 )
      print("Finish task", masks.shape, len(ids), len(meta_data), len(stats_)  )
      return masks, ids, meta_data, stats_


  train_labels = glob.glob('../data/train_label/*.png')
  jobs = []

  for label in train_labels[:5]:
      result = get_masks_from_image.remote(label)
      jobs.append(result)

  t6 = time.time() 
  result = ray.get(jobs)
  t6_ = time.time()
  print("Getting Results from ray jobs takes:", t6_- t6 )
  # print(result)
  end_time = time.time()
  print("This sequence holds:", end_time-start_time )