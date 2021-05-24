import requests
import pandas as pd
import urllib.request
import re
import time


from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from PIL import Image

import numpy as np
import pandas as pd
import os
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm



train_image_file_list = pd.read_csv('data/file_list.csv',).iloc[:, 0].to_list()


efficientnetb7_model = EfficientNetB7(weights='imagenet', include_top=True)
efficientnetb7_model = Model(inputs=efficientnetb7_model.input, outputs=efficientnetb7_model.get_layer('avg_pool').output)

def extract_efficientnetb7_model_from_list(image_file_list, mode='sum'):
    all_feature = []
    for idx, eachFile in tqdm(enumerate(image_file_list)):
#         if ((idx % 100) ==0 ):
#             print("process %d/%d file" % (idx+1, len(image_file_list)))
    
        img = image.load_img(eachFile, target_size=(600, 600))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        conv_features = efficientnetb7_model.predict(x)
        if (mode == 'sum'):
#             conv_features = np.sum(conv_features, axis=(1,2))
            conv_features = conv_features/np.linalg.norm(conv_features) # normalzie features
        else:
            print('extract_vgg_conv_feature_from_list: Wrong mode as input')
            break
        all_feature.append(conv_features)
    return all_feature

def extract_efficientnetb7_model_from_image(image):
    all_feature = []
    
    x = cv2.resize(image, (600,600), interpolation = cv2.INTER_AREA)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    conv_features = efficientnetb7_model.predict(x)
    conv_features = conv_features/np.linalg.norm(conv_features) # normalzie features
    all_feature.append(conv_features)
    
    return all_feature


def cal_vec_dist(vec1, vec2):
    '''
    Description: calculate the Euclidean Distance of two vectors
    '''
    return np.linalg.norm(vec1 - vec2)


train_all_feature_efficientnetb7_model = np.load('data/train_all_feature_efficientnetb7_model.npy')



def search_image(image, return_top = 5):
    test_img_feature = extract_efficientnetb7_model_from_image(image)

    top_return = 5
    dist_list = []
    final_filename = []
    final_path = []
    for eachpic in range(len(train_all_feature_efficientnetb7_model)):
        dist = cal_vec_dist(test_img_feature, train_all_feature_efficientnetb7_model[eachpic])
        dist_list.append(dist)
        final_path.append(train_image_file_list[eachpic])
        final_filename.append(train_image_file_list[eachpic].split('/')[2])
    result = pd.DataFrame({'dist': dist_list, 'filename': final_filename, 'path': final_path})
    result = result.sort_values(by='dist').drop_duplicates('filename')
    return result[:return_top].path.tolist()


def search_image_path(image, return_top = 5):
    test_img_feature = extract_efficientnetb7_model_from_list(image)#[0]

    top_return = 5
    dist_list = []
    final_filename = []
    final_path = []
    for eachpic in range(len(train_all_feature_vgg_conv)):
        dist = cal_vec_dist(test_img_feature, train_all_feature_vgg_conv[eachpic])
        dist_list.append(dist)
        final_path.append(train_image_file_list[eachpic])
        final_filename.append(train_image_file_list[eachpic].split('/')[2])
    result = pd.DataFrame({'dist': dist_list, 'filename': final_filename, 'path': final_path})
    result = result.sort_values(by='dist').drop_duplicates('filename')
    return result[:return_top].path.tolist()

