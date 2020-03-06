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


#F:\deepak\Thesis_test_code\ann\images\train\angry
IMG_DIR = 'F:/deepak/Thesis_test_code/ann/images/train/angry/*'
angry = glob.glob(IMG_DIR)
print("number of images in angry = "+ str(len(angry)))

angry_FolderName = [str(i.split("\\")[0])+"/" for i in angry]
angry_ImageName = [str(i.split("\\")[1]) for i in angry]
angry_Emotion = [["Angry"]*len(angry)][0]
angry_label =[1]*len(angry)

len(angry_FolderName) , len(angry_ImageName), len(angry_Emotion), len(angry_label)


df_angry = pd.DataFrame()
df_angry["FolderName"] = angry_FolderName
df_angry["ImageName"] = angry_ImageName
df_angry["Emotion"] = angry_Emotion
df_angry["Labels"] = angry_label
df_angry.head()

print(df_angry.head())




IMG_DIR1 = 'F:/deepak/Thesis_test_code/ann/images/train/disgust/*'
disgust = glob.glob(IMG_DIR1)
print("number of images in disgust = "+ str(len(disgust)))

disgust_FolderName = [str(i.split("\\")[0])+"/" for i in disgust]
disgust_ImageName = [str(i.split("\\")[1]) for i in disgust]
disgust_Emotion = [["Disgust"]*len(disgust)][0]
disgust_label =[2]*len(disgust)

len(disgust_FolderName) , len(disgust_ImageName), len(disgust_Emotion), len(disgust_label)


df_disgust = pd.DataFrame()
df_disgust["FolderName"] = disgust_FolderName
df_disgust["ImageName"] = disgust_ImageName
df_disgust["Emotion"] = disgust_Emotion
df_disgust["Labels"] = disgust_label
df_disgust.head()

print(df_disgust.head())


IMG_DIR2 = 'F:/deepak/Thesis_test_code/ann/images/train/fear/*'
fear = glob.glob(IMG_DIR2)
print("number of images in fear = "+ str(len(fear)))

fear_FolderName = [str(i.split("\\")[0])+"/" for i in fear]
fear_ImageName = [str(i.split("\\")[1]) for i in fear]
fear_Emotion = [["Fear"]*len(fear)][0]
fear_label =[3]*len(fear)

len(fear_FolderName) , len(fear_ImageName), len(fear_Emotion), len(fear_label)


df_fear = pd.DataFrame()
df_fear["FolderName"] = fear_FolderName
df_fear["ImageName"] = fear_ImageName
df_fear["Emotion"] = fear_Emotion
df_fear["Labels"] = fear_label
df_fear.head()

print(df_fear.head())



IMG_DIR3 = 'F:/deepak/Thesis_test_code/ann/images/train/happy/*'
happy = glob.glob(IMG_DIR3)
print("number of images in happy = "+ str(len(happy)))

happy_FolderName = [str(i.split("\\")[0])+"/" for i in happy]
happy_ImageName = [str(i.split("\\")[1]) for i in happy]
happy_Emotion = [["Happy"]*len(happy)][0]
happy_label =[4]*len(happy)

len(happy_FolderName) , len(happy_ImageName), len(happy_Emotion), len(happy_label)


df_happy = pd.DataFrame()
df_happy["FolderName"] = happy_FolderName
df_happy["ImageName"] = happy_ImageName
df_happy["Emotion"] = happy_Emotion
df_happy["Labels"] = happy_label
df_happy.head()

print(df_happy.head())



IMG_DIR4 = 'F:/deepak/Thesis_test_code/ann/images/train/neutral/*'
neutral = glob.glob(IMG_DIR4)
print("number of images in neutral = "+ str(len(neutral)))

neutral_FolderName = [str(i.split("\\")[0])+"/" for i in neutral]
neutral_ImageName = [str(i.split("\\")[1]) for i in neutral]
neutral_Emotion = [["Neutral"]*len(neutral)][0]
neutral_label =[5]*len(neutral)

len(neutral_FolderName) , len(neutral_ImageName), len(neutral_Emotion), len(neutral_label)


df_neutral = pd.DataFrame()
df_neutral["FolderName"] = neutral_FolderName
df_neutral["ImageName"] = neutral_ImageName
df_neutral["Emotion"] = neutral_Emotion
df_neutral["Labels"] = neutral_label
df_neutral.head()

print(df_neutral.head())




IMG_DIR5 = 'F:/deepak/Thesis_test_code/ann/images/train/sad/*'
sad = glob.glob(IMG_DIR5)
print("number of images in sad = "+ str(len(sad)))

sad_FolderName = [str(i.split("\\")[0])+"/" for i in sad]
sad_ImageName = [str(i.split("\\")[1]) for i in sad]
sad_Emotion = [["Sad"]*len(sad)][0]
sad_label =[6]*len(sad)

len(sad_FolderName) , len(sad_ImageName), len(sad_Emotion), len(sad_label)


df_sad = pd.DataFrame()
df_sad["FolderName"] = sad_FolderName
df_sad["ImageName"] = sad_ImageName
df_sad["Emotion"] = sad_Emotion
df_sad["Labels"] = sad_label
df_sad.head()

print(df_sad.head())



IMG_DIR6 = 'F:/deepak/Thesis_test_code/ann/images/train/surprise/*'
surprise = glob.glob(IMG_DIR6)
print("number of images in surprise = "+ str(len(surprise)))

surprise_FolderName = [str(i.split("\\")[0])+"/" for i in surprise]
surprise_ImageName = [str(i.split("\\")[1]) for i in surprise]
surprise_Emotion = [["Surprise"]*len(surprise)][0]
surprise_label =[7]*len(surprise)

len(surprise_FolderName) , len(surprise_ImageName), len(surprise_Emotion), len(surprise_label)


df_surprise = pd.DataFrame()
df_surprise["FolderName"] = surprise_FolderName
df_surprise["ImageName"] = surprise_ImageName
df_surprise["Emotion"] = surprise_Emotion
df_surprise["Labels"] = surprise_label
df_surprise.head()

print(df_surprise.head())







frames= [df_angry, df_disgust , df_fear , df_happy , df_neutral , df_sad , df_surprise]
final = pd.concat(frames)
print(final.shape)

final.reset_index(inplace = True , drop = True)
final = final.sample(frac = 1.0)
final.reset_index(inplace = True , drop = True)
print(final.head())


df_train_data , df_test = train_test_split(final , stratify = final["Labels"], test_size = 0.153973)
df_train,df_cv = train_test_split(df_train_data , stratify = df_train_data["Labels"], test_size = 0.153973)
print(df_train.shape)
print(df_cv.shape)
print(df_test.shape)


'''
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

def face_det_crop_resize(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]  #cropping the face in image
        cv2.imwrite(img_path, cv2.resize(face_clip, (350, 350)))  #resizing image then saving it

for i, d in df_train.iterrows():
    img_path = os.path.join(d["FolderName"], d["ImageName"])
    face_det_crop_resize(img_path)

for i, d in df_cv.iterrows():
    img_path = os.path.join(d["FolderName"], d["ImageName"])
    face_det_crop_resize(img_path)

for i, d in df_test.iterrows():
    img_path = os.path.join(d["FolderName"], d["ImageName"])
    face_det_crop_resize(img_path)

'''

df_train.reset_index(inplace = True , drop = True)
df_train.to_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_train.pkl")

df_cv.reset_index(inplace = True , drop = True)
df_cv.to_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_cv.pkl")

df_test.reset_index(inplace = True , drop = True)
df_test.to_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_test.pkl")

df_trainn = pd.read_pickle("F:/deepak/Thesis_test_code//ann/pickles/df_train.pkl")
print(df_trainn.head())
print(df_trainn.shape)

df_cvv = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_cv.pkl")
print(df_cvv.head())
print(df_cvv.shape)

df_testt = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_test.pkl")
print(df_testt.head())
print(df_testt.shape)


