# # Start from here

# In[1]:


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
import matplotlib.pyplot as plt
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image


# # Run a pre-trained detectron2 model

# We first download an image from the COCO dataset:

# In[200]:


def display_img(img,cmap=None):
    fig = plt.figure(figsize = (12,12))
    plt.axis(False)
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)


# In[205]:


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set threshold for this model
cfg.MODEL.DEVICE = 'cpu'
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


# In[206]:


def detect_object(image):
    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_with_label = out.get_image()[:, :, ::-1]

    mask = outputs["instances"].pred_masks.to("cpu").numpy()
    mask = mask.astype(int)
    mask=np.moveaxis(mask, 0, -1)
    boxes=outputs["instances"].pred_boxes.tensor.cpu().numpy()
    boxes = boxes.astype(int)

    class_no = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist()
    class_name = []
    for i, _ in enumerate(class_no):
        class_name.append(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[_] + " "
                          + str(round(scores[i] * 100)) + '%')

    output_list = []

    for i in range(mask.shape[2]):
        temp = image.copy()
        for j in range(temp.shape[2]):
          temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
          (x, y, w, h) = (boxes[i][0], boxes[i][1],boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1])
          temp_nobg=temp[y:y+h,x:x+w]+255
        output_list.append(temp_nobg)
    
    return(image_with_label, class_name, output_list)



