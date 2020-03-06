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


train = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_train.pkl")
cv = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_cv.pkl")
test = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_test.pkl")

train_batch_pointer = 0
cv_batch_pointer = 0
test_batch_pointer = 0

train_labels = pd.get_dummies(train["Labels"]).as_matrix()
print(train.shape)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = train.iloc[train_batch_pointer + i]["FolderName"]
        path2 = train.iloc[train_batch_pointer + i]["ImageName"]
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0
        batch_images.append(read_image_final)
        batch_labels.append(train_labels[train_batch_pointer + i])   
    train_batch_pointer += batch_size
    return np.array(batch_images), np.array(batch_labels)



model = VGG16(weights='imagenet',include_top= False)
SAVEDIR = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Bottleneck_train/"
SAVEDIR_LABELS = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Train_Labels/"
batch_size1 = 10
for i in range(int(len(train)/batch_size1)):
    x,y = LoadTrainBatch(batch_size1)
    print("Batch {} loaded".format(i+1))

    np.save(os.path.join(SAVEDIR_LABELS,"bottleneck_labels_{}".format(i+1)),y)

    print("Creating bottleneck feature for batch {}".format(i+1))
    bottleneck_Features = model.predict(x)
    np.save(os.path.join(SAVEDIR,"bottleneck_{}".format(i+1)),bottleneck_Features)
    print("Bottleneck features for batch {} created and Saved\n".format(i+1))



cv_labels = pd.get_dummies(cv["Labels"]).as_matrix()
print(cv.shape)


def LoadCvBatch(batch_size1):
    global cv_batch_pointer
    batch_images1 = []
    batch_labels1 = []
    for i in range(batch_size1):
        path11 = train.iloc[cv_batch_pointer + i]["FolderName"]
        path21 = train.iloc[cv_batch_pointer + i]["ImageName"]
        read_image1 = cv2.imread(os.path.join(path11, path21))
        read_image_final1 = read_image1/255.0
        batch_images1.append(read_image_final1)
        batch_labels1.append(cv_labels[cv_batch_pointer + i])   
    cv_batch_pointer += batch_size1
    return np.array(batch_images1), np.array(batch_labels1)


model = VGG16(weights='imagenet',include_top= False)
SAVEDIR1 = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Bottleneck_cv/"
SAVEDIR_LABELS1 = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Cv_Labels/"
batch_size2 = 10
for i in range(int(len(cv)/batch_size2)):
    x1,y1 = LoadCvBatch(batch_size2)
    print("Batch {} loaded".format(i+1))

    np.save(os.path.join(SAVEDIR_LABELS1,"bottleneck_labels_{}".format(i+1)),y1)

    print("Creating bottleneck feature for batch {}".format(i+1))
    bottleneck_Features1 = model.predict(x1)
    np.save(os.path.join(SAVEDIR1,"bottleneck_{}".format(i+1)),bottleneck_Features1)
    print("Bottleneck features for batch {} created and Saved\n".format(i+1))




test_labels = pd.get_dummies(test["Labels"]).as_matrix()
print(test.shape)


def LoadTestBatch(batch_size2):
    global test_batch_pointer
    batch_images2 = []
    batch_labels2 = []
    for i in range(batch_size2):
        path12 = train.iloc[test_batch_pointer + i]["FolderName"]
        path22 = train.iloc[test_batch_pointer + i]["ImageName"]
        read_image1 = cv2.imread(os.path.join(path12, path22))
        read_image_final2 = read_image1/255.0
        batch_images2.append(read_image_final2)
        batch_labels2.append(test_labels[test_batch_pointer + i])   
    test_batch_pointer += batch_size2
    return np.array(batch_images2), np.array(batch_labels2)


model = VGG16(weights='imagenet',include_top= False)
SAVEDIR2 = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Bottleneck_test/"
SAVEDIR_LABELS2 = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Test_Labels/"
batch_size2 = 10
for i in range(int(len(cv)/batch_size2)):
    x2,y2 = LoadTestBatch(batch_size2)
    print("Batch {} loaded".format(i+1))

    np.save(os.path.join(SAVEDIR_LABELS2,"bottleneck_labels_{}".format(i+1)),y2)

    print("Creating bottleneck feature for batch {}".format(i+1))
    bottleneck_Features2 = model.predict(x2)
    np.save(os.path.join(SAVEDIR2,"bottleneck_{}".format(i+1)),bottleneck_Features2)
    print("Bottleneck features for batch {} created and Saved\n".format(i+1))


print(bottleneck_Features2.shape)




