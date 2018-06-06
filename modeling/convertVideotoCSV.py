#!/usr/bin/python


## The script for converting prediction into csv format.
# This script accepts two params as input:
# 1. $prediciton: Prediction folder which contains prediction results;
# 2. VideoList which contains frame names of the video to be converted.
# For example: python convertVideotoCSV.py  $predition \
# road0*_cam_*_video_*_image_list_test.txt 

# To run this script, make sure that your results contain text files in 
# the prediction folder.
# (one for each test set image) with the content:
#   relPathPrediction1 labelIDPrediction1 confidencePrediction1
#   relPathPrediction2 labelIDPrediction2 confidencePrediction2
#   relPathPrediction3 labelIDPrediction3 confidencePrediction3
#   ...
#
# - The given paths "relPathPrediction" point to images that contain
# binary masks for the described predictions, where any non-zero is
# part of the predicted instance. 
# - The label IDs "labelIDPrediction" specify the class of that mask,
# encoded as defined in labels.py. Note that the regular ID is used,
# not the train ID.
# - The field "confidencePrediction" is a float value that assigns a
# confidence score to the mask.
#
# Note that this tool creates a csv file named "predicition.csv"

# python imports
from __future__ import print_function
import os, sys
import fnmatch
from copy import deepcopy
from PIL import Image
import numpy as np
import hashlib
import time


def printError(s):
    print(s)
    
###
### A command to use this script would be:
### python convertVideotoCSV.py  test_masks/ list_test.txt
###
    
###################################
# PLEASE READ THESE INSTRUCTIONS!!!
###################################
# Provide the prediction file for the given ground truth file.
# Please read the instructions above for a description of
# the result file.
#
# Within the prediction folder, a matching prediction file is recursively searched.
# A file matches, if the filename follows the pattern
# 123456_123456789_Camera_[5,6]_instanceIds.txt
# for a ground truth filename
# 123456_123456789_Camera_[5,6]_instanceIds.png
def getPrediction( groundTruthFile , args ):
    # determine the prediction path, if the method is first called
    if not args.predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(args.predictionPath):
            walk.append( (root,filenames) )
        args.predictionWalk = walk
    csName = groundTruthFile.split('/')[-1].split('\\')[-1].split('_')[0:-1]
    filePattern = "{}_{}_{}_{}_instanceIds.txt".format( csName[0] , csName[1] , csName[2], csName[3] )
    predictionFile = None
    for root, filenames in args.predictionWalk:
         
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError("Found multiple predictions for ground truth {}".format(groundTruthFile))

    if not predictionFile:
        # import pdb ; pdb.set_trace()
        printError("Found no prediction for ground truth {}".format(groundTruthFile))

    return predictionFile


######################
# Parameters
######################


# A dummy class to collect all bunch of data
class CArgs(object):
    pass
# And a global object of that class
args = CArgs()

args.quiet              = False
args.predictionPath = None
args.predictionWalk = None
# output csv file path that should be modified by user
args.csvName = "./prediction.csv"

def processImage(predictionList, args, list_index, filename):
    csv = open(args.csvName,'a', encoding='utf-8')
    print("Job Stars id:", list_index)
    filename = filename.replace("\\","/")
    print(filename)
    name = filename.split('/')[-1]
    m1 = hashlib.md5()

    filenameMD5 = m1.update(name.encode('utf-8'))

    predicitionFile = open(predictionList[list_index], "r", encoding='utf-8')
    predictionlines = predicitionFile.readlines()
    for predictionline in predictionlines:
        predictionInfo = predictionline.split(' ')
        img = Image.open(args.predictionPath+predictionInfo[0])
        predictionname = predictionInfo[0].split('/')[-1]
        m2=hashlib.md5()
        predictionnameMD5 = m2.update(predictionname.encode('utf-8'))
        InstanceMap = (np.array(img) > 0) * 200
        check = 0
        csv.write("{},".format(m1.hexdigest()))
        csv.write("{},".format(predictionInfo[1]))
        csv.write("{},".format(predictionInfo[2].split('\n')[0]))
        begin = 0
        length = 0
        idmap1d = np.reshape(InstanceMap==200,(-1))
        InstanceIds = np.unique(InstanceMap)
        Totalcount = np.sum(idmap1d)
        csv.write("{},".format(Totalcount))
        find = False
        for index in range(idmap1d.shape[0]):
            if find:
                if idmap1d[index]:
                    length = length+1
                else:
                    csv.write("{} {}|".format(begin,length))
                    check = check + length
                    length = 0
                    find = False
            else:
                if idmap1d[index]:
                    begin = index
                    length = 1
                    find = True
            if index == idmap1d.shape[0]-1 and find:
                csv.write("{} {}|".format(begin,length))
                check = check + length
                length = 0
                find = False
        csv.write("\n")
        if Totalcount == check:
            print("LabelId = {},success.".format(predictionInfo[1]))
        else:
            print("failed!{}vs{}.".format(Totalcount,check))
        check = 0
    print("Job End id:", list_index)
    return "ok"

def convertImages(predictionList ,groundTruthList, args):
    from multiprocessing import Pool
    pool = Pool(8)
    csv = open(args.csvName,'a', encoding='utf-8')
    csv.close()
    jobs = []

    for list_index,filename in enumerate(groundTruthList):
        if predictionList[list_index] == None:
            print("Id None: ", list_index, " ", groundTruthList[list_index] )
            continue
        job = pool.apply_async(processImage, (predictionList, args, list_index, filename) )
        jobs.append(job)
        # processImage(predictionList, args, list_index, filename)
    for job in jobs:
        p_result = job.get(timeout=100000)

# The main method
def main(argv):
    global args
    args.predictionPath = argv[0]
    argv[1] = open(argv[1]).read()
    videoname = argv[1].split('/')[-1].split('.')[0]
    print('videoname:'+videoname)
    groundTruthImgList = []
    predictionImgList = []
    ######read list from argv[1] 
    groundTruthList = open(argv[1].split('\n')[0].split('\r')[0], "r", encoding='utf-8')     
    groundTruthListlines = groundTruthList.readlines()
    # import pdb ; pdb.set_trace()
    for groundTruthListline in groundTruthListlines:
        gtfilename = groundTruthListline.split('\n')[0].split('\t')[0]
        gtfilename = gtfilename.replace('ColorImage','Label')
        gtfilename = gtfilename.replace('.jpg','_instanceIds.png')
        gtfilename = gtfilename.replace('\\','/')
        gtfilename = gtfilename.replace('//','/')
        groundTruthImgList.append(gtfilename)
    for gt in groundTruthImgList:
        predictionImgList.append( getPrediction(gt,args))
    
    convertImages(predictionImgList, groundTruthImgList, args)
    return

# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
