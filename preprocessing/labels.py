import numpy as np 
import pandas as pd
"""
This is going to be the job thar will be
share accross workers.
This contain other functions.
"""

def get_masks_from_image(image_id):
    """
    image_id: image path
    returns instances ids of objects
    """

    image = cv2.imread(image_id, cv2.IMREAD_UNCHANGED)
    ids = []
    intances_ids = np.unique(image)
    intances_ids = np.delete(intances_ids, np.where(intances_ids == 255))

    masks_shape = (image.shape[0], image.shape[1], intances_ids.shape[0])
    masks = np.zeros(masks_shape, dtype=np.bool)

    for i in range(intances_ids.shape[0]):
        class_id = int(intances_ids[i]/1000)
        ids.append(class_id)

        masks[:, :, i] = (image == intances_ids[i])

    print("Finish task")
    return ids

if __name__ == '__main__':
  """
  This is the driver program
  """
  from multiprocessing import Pool
  import time
  import cv2
  import glob
  pool = Pool(processes=8)
  jobs = []
  train_labels = glob.glob('../data/train_label/*.png')
  start_time = time.time()

  
  for label in train_labels[:100]:
      result = pool.apply_async(get_masks_from_image, (label,))
      jobs.append(result)
  
  t6 = time.time()
  for job in jobs:
    _ = job.get(timeout=20)
  t6_ = time.time()
  

  print("Getting Results from ray jobs takes:", t6_- t6 )
  end_time = time.time()
  print("Entire Process Takes:", end_time-start_time )