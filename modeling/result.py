import shutil
import glob
import pandas as pd
import os
import cv2
from multiprocessing import Pool
import time
from skimage.io import imsave
import shutil

class Prediction():

  def __init__(self, dataset, mask_prediction_path, data_dir, cores=1):
    self.data_dir = data_dir
    self.dataset = dataset
    self.mask_prediction_path = mask_prediction_path
    self.core = cores
    self.mapping_md5_images_names = glob.glob(self.data_dir + "/test_video_list_and_name_mapping/list_test_mapping/*.txt")
    self.test_mask_dir = mask_prediction_path

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
    """
    It receives predictions, creates a csv with results
    and return datapoints
    """
    data = []
    with open(self.test_mask_dir + image_name + "_instanceIds.txt", 'a') as f:
      
      for i in range(len(r['scores'])):
          mask_idx = i
          confidence = r['scores'][i]
          label_id = r['class_ids'][i]
          mask = r['masks'][:, :, i]
          x1, x2, y1, y2 = r['rois'][i]

          data_point = {}
          old_id = self.dataset.CLASS_ID_TO_NAME[label_id]['old_id']
          mask_path = "{}{}_{}.jpg".format( self.mask_prediction_path, image_name, str(mask_idx) )
          mask_name = mask_path
          imsave(mask_name, mask)

          data_point['ImageId'] = image_name
          data_point['LabelId'] = old_id
          data_point['Confidence'] = confidence
          data_point['x1'] = x1
          data_point['x2'] = x2
          data_point['y1'] = y1
          data_point['y2'] = y2
          data_point['mask_path'] = mask_name

          line = "{} {} {}\n".format(mask_name, data_point['LabelId'], data_point['Confidence'])
          f.write(line)
          data.append(data_point)
    f.close()
    return data


  def generate_results(self, model):
    """
    Process in parallel process_result and join results
    """
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
        p_result = job.get(timeout=100)
        test_results = test_results + p_result
    t3_ = time.time()
    print("Time for all takes:", t3_ - t3)

  def organize_prediction_masks(self):
    for i in range( len(self.mapping_md5_images_names) ):
      frames = pd.read_csv(self.mapping_md5_images_names[i], sep='\t', header=None)
      for j in range(frames.shape[0]):
          image_id = frames.iloc[j][0]
          path = frames.iloc[j][1].split("\\")[-1].split(".")[-2]
          preds = glob.glob(self.test_mask_dir + image_id + "*jpg")
          camera_folder = self.test_mask_dir + path

          if not os.path.exists(camera_folder + "_instanceIds.txt"):
            shutil.move(self.test_mask_dir + image_id + "_instanceIds.txt", camera_folder + "_instanceIds.txt" )

          if os.path.exists(camera_folder):
              print("Warning already exists: ", camera_folder)
          else:
              os.makedirs(camera_folder)
          for p_mask in preds:
              image_instance_id = p_mask.split("/")[-1]
              shutil.move(p_mask, camera_folder + "/" + image_instance_id )

