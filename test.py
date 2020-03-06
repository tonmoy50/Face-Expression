import os
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from PIL import Image
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,load_model
from keras.applications import VGG16
from sklearn.metrics import accuracy_score, confusion_matrix


model = load_model("F:/deepak/Thesis_test_code/ann/Model_Save/model.h5")
predicted_labels = []
true_labels = []
batch_size = 10

Test = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_test.pkl")

total_files = 370  #here, I have added 2 because there are 30 files in Test_Humans
for i in range(1, total_files, 1):
    img_load = np.load("F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Bottleneck_test/bottleneck_{}.npy".format(i))
    img_label = np.load("F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Test_Labels/bottleneck_labels_{}.npy".format(i))
    img_bundle = img_load.reshape(img_load.shape[0], img_load.shape[1]*img_load.shape[2]*img_load.shape[3])
    for j in range(img_bundle.shape[0]):
        img = img_bundle[j]
        img = img.reshape(1, img_bundle.shape[1])
        pred = model.predict(img)
        predicted_labels.append(pred[0].argmax())
        true_labels.append(img_label[j].argmax())
acc = accuracy_score(true_labels, predicted_labels)
print("Accuracy on Test Data = {}%".format(np.round(float(acc*100), 2)))