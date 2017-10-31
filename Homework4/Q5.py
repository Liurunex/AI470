"""
Q5: Determining Traffic Signals and Proximity warnings with Tensorflow and OpenCV

For this last part you will be working with two popular and powerful frameworks: TensorFlow (Deep Learning) and OpenCV (Computer Vision). 
You will be using the recently released TensorFlow Object Detection API to detect cars, buses, trucks and traffic lights.
Tensorflow Object Detection API provides access to the pre-trained models for out-of-the-box inference purposes. 
There, you will learn how the frameworks work, culminating in the development of algorithms for further image analysis with the aim
to detect traffic signal lights and display proximity warnings. 

"""

import sys
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import label_map_util
import visualization_utils as vis_util
from os import listdir
from os.path import isfile, join

# This is needed to display the images.
get_ipython().magic('matplotlib inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = PATH_NAME + MODEL_NAME + '\\frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(PATH_NAME,'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
  
PATH_TO_TEST_IMAGES_DIR = 'CPSC470/datasets/test_images/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
                                  
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      print('visualizing: ' + image_path)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      
      
      
import cv2      

# OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. 
# OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception 
# in the commercial products.   

# There are more than 150 color-space conversion methods available in OpenCV. 
# But we will look into only two which are most widely used ones, BGR \leftrightarrow Gray and BGR \leftrightarrow HSV.

# For color conversion, we use the function cv2.cvtColor(input_image, flag) where flag determines the type of conversion.

# For BGR \rightarrow Gray conversion we use the flags cv2.COLOR_BGR2GRAY. 
# Similarly for BGR \rightarrow HSV, we use the flag cv2.COLOR_BGR2HSV. 
# To get other flags, just run following commands in your Python terminal :
    
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)
    
img = cv2.imread('CPSC470/datasets/test_images/opencv-logo2.jpg')

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)

cv2.imshow('frame',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
    
#########################################################################################################
# TODO: Implement the following function using OpenCV to estimate the state of the Traffic light signal #                                      #
#########################################################################################################

def display_traffic_light_signal(image_np, boxes):
  traffic_signal = "UNDEF"
  im_height, im_width, channels = image_np.shape
  # Insert your OpenCV code to detect the Traffic light signal        
  mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
  mid_y = (boxes[0][i][0]+boxes[0][i][2])/2   
  cv2.putText(image_np, traffic_signal, (int(mid_x*im_width),int(mid_y*im_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
  cv2.putText(image_np, 'DO not Drive!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 3)

####################################################################################            
# Implement the following function using OpenCV to estimate other vehicle distances#
####################################################################################  
  
def display_collision_warning(image_np, boxes):
   drv_warning = 'Collision warning'
   # Insert your OpenCV code to estimate other vehicle distances and display 'Collision warning' message
   cv2.putText(image_np, drv_warning, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 3)                      

PATH_TO_TEST_IMAGES_DIR = 'CPSC470/datasets/Udacity'
TEST_IMAGE_PATHS = [f for f in listdir(PATH_TO_TEST_IMAGES_DIR) if isfile(join(PATH_TO_TEST_IMAGES_DIR, f))]
                                  
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

video = cv2.VideoWriter('Q5_video.avi',-1,1,(960,540))
      
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open( os.path.join(PATH_TO_TEST_IMAGES_DIR, image_path))
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.              
      print('visualizing: ' + image_path)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
    
      for i,b in enumerate(boxes[0]):
        # traffic light
        if classes[0][i] == 10:   
           if scores[0][i] >= 0.5:
               display_traffic_light_signal(image_np, boxes)
               
       # car, bus, truck
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
           if scores[0][i] >= 0.5:
               display_collision_warning(image_np, boxes)
      
      image_np_res = cv2.resize(image_np, (960, 540))  
      image_np_res_RGB_img = cv2.cvtColor(image_np_res, cv2.COLOR_BGR2RGB)
      cv2.imshow('window',image_np_res_RGB_img)
      video.write(image_np_res_RGB_img)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          video.release()
          break

cv2.destroyAllWindows()
video.release()  


