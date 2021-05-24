#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input

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


# In[2]:


ikea_data = pd.read_csv('data/ikea_products.csv', index_col=False)
hipvan_data = pd.read_csv('data/hipvan_products.csv', index_col=False)

ikea_category = list(ikea_data.category_name.unique())
hipvan_category = list(hipvan_data.category.unique())
all_category = ikea_category + hipvan_category

# Prepare the training folder and image file list
image_folder_name = ['ikea_image', 'hipvan_image']
image_file_path = []

for folder_name in image_folder_name:
    for subfolder in all_category:
        try:
            file_list = os.listdir(folder_name + '/' + subfolder)
            for file in file_list:
                image_file_path.append(folder_name + '/' + subfolder + '/' + file)
        except:
            pass

image_file_path_df = pd.DataFrame({'image_file_path': image_file_path})
image_file_path_df['label'] = image_file_path_df.image_file_path.str.split(pat = '/').str[1]

# train_image_file_list, test_image_file_list, train_label, test_label = train_test_split(image_file_path_df.image_file_path, image_file_path_df.label, test_size=0.2, random_state=42, stratify = image_file_path_df.label)

train_image_file_list = image_file_path_df.image_file_path
train_label = image_file_path_df.label


train_image_file_list = train_image_file_list.to_list()
# test_image_file_list = test_image_file_list.to_list()




# In[6]:


vgg_model = vgg16.VGG16(weights='imagenet', include_top=True)
# vgg_model.summary()

# Define VGG extracting block5_conv3 layer coefficients
vgg_model_extract_blk5conv3 = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_conv3').output)

# display function to show image
def display_img(img,cmap=None):
    fig = plt.figure(figsize = (12,12))
    plt.axis(False)
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    
def extract_vgg_conv_feature_from_list(image_file_list, mode='sum'):
    all_feature = []
    for idx, eachFile in enumerate(image_file_list):
        if ((idx % 100) ==0 ):
            print("process %d/%d file" % (idx+1, len(image_file_list)))
            
        img = image.load_img(eachFile, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        conv_features = vgg_model_extract_blk5conv3.predict(x)
        if (mode == 'sum'):
            conv_features = np.sum(conv_features, axis=(1,2))
            conv_features = conv_features/np.linalg.norm(conv_features) # normalzie features
        else:
            print('extract_vgg_conv_feature_from_list: Wrong mode as input')
            break
        all_feature.append(conv_features)
    return all_feature

def extract_vgg_conv_feature_from_image(image):
    all_feature = []
    x = Image.fromarray(image).resize(size=(224,224))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    conv_features = vgg_model_extract_blk5conv3.predict(x)

    conv_features = np.sum(conv_features, axis=(1,2))
    conv_features = conv_features/np.linalg.norm(conv_features) # normalzie features
    all_feature.append(conv_features)
    
    return all_feature

def cal_vec_dist(vec1, vec2):
    '''
    Description: calculate the Euclidean Distance of two vectors
    '''
    return np.linalg.norm(vec1 - vec2)


# In[7]:


# # # Uncomment followings to re-calculate all features
# # Extract features from training images
# train_all_feature_vgg_conv = extract_vgg_conv_feature_from_list(train_image_file_list, 'sum')
# np.save('data/train_all_feature_vgg_conv.npy', train_all_feature_vgg_conv)
# print("Train_all_feature_vgg_conv: (%d, %d)" % (len(train_all_feature_vgg_conv), train_all_feature_vgg_conv[0].shape[1]))


# In[132]:


# Use the pre-calculated features provided in the workshop

train_all_feature_vgg_conv = np.load('data/train_all_feature_vgg_conv.npy')
# print("Train_all_feature_vgg_conv: (%d, %d)" % (len(train_all_feature_vgg_conv), train_all_feature_vgg_conv[0].shape[1]))



# In[221]:


def search_image(image, return_top = 5):
    test_img_feature = extract_vgg_conv_feature_from_image(image)#[0]

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


# In[222]:


def search_image_path(image, return_top = 5):
    test_img_feature = extract_vgg_conv_feature_from_list(image)#[0]

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







