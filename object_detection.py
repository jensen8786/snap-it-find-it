#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Load the class labels
LABELS = open("data/object_detection_classes_yolov3.txt").read().strip().split("\n")


# In[9]:


weightsPath = os.path.join("data/yolov3.weights")
configPath = os.path.join("data/yolov3.cfg")


# In[10]:


# Loading the neural network framework Darknet (YOLO was created based on this framework)
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)


# In[24]:


# Create the function which predict the frame input
def detect_object(image):
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    image2 = image.copy()
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    (H, W) = image.shape[:2]
    
    # determine only the "ouput" layers name which we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image2, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    output_list = []
    boxes = []
    coord = []
    confidences = []
    classIDs = []
    classes = []
    box_label_list=[]
    threshold = 0.2
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            # confidence type=float, default=0.5
            if confidence > threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)

    # ensure at least one detection exists
    if len(idxs) > 0:
      
      # loop over the indexes we are keeping
      for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
           
        coord.append([x,y,w,h])

        #for image cropping
        output_list.append(image.copy()[y:y+h, x:x+w])

        #prepare list of labels 
        box_label_list.append(LABELS[classIDs[i]])

      #take care of duplicates
      dups = {}
      for j, val in enumerate(box_label_list):
        if val not in dups:
          #store index of first occurence and occurence value
          dups[val] = [j, 1]
        else:
          #special case for first occurence
          if dups[val][1] ==1:
            box_label_list[dups[val][0]] += str(dups[val][1])

          #increment occurence value, index value doesn't matter anymore
          dups[val][1] += 1
            
          #use stored occurence value
          box_label_list[j] += str(dups[val][1])

      for k in range(len(box_label_list)):
        # draw a bounding box rectangle and label on the image
        x = coord[k][0]
        y = coord[k][1]
        w = coord[k][2]
        h = coord[k][3]
        color = (255,0,0)
        cv2.rectangle(image2, (x, y), (x + w, y + h), color, 2)
        text = box_label_list[k]
        cv2.putText(image2, text, (x +15, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        classes.append(text)
        
    return image2, coord, classes, output_list


# In[ ]:




