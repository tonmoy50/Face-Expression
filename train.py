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


no_of_classes = 7

train = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_train.pkl")
cv = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_cv.pkl")
test = pd.read_pickle("F:/deepak/Thesis_test_code/ann/pickles/df_test.pkl")

def model(input_shape):
    model = Sequential()
        
    model.add(Dense(512 , activation='relu', input_dim = input_shape))
    model.add(Dropout(0.1))
    
    model.add(Dense(256, activation='relu'))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense( no_of_classes, activation='softmax')) 
    
    return model


SAVEDIR_TRAIN = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Bottleneck_train/"
SAVEDIR_TRAIN_LABELS = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Train_Labels/"

SAVEDIR_CV = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Bottleneck_cv/"
SAVEDIR_CV_LABELS = "F:/deepak/Thesis_test_code/ann/Bottleneck_Features/Cv_Labels/"


SAVER = "F:/deepak/Thesis_test_code/ann/Model_Save/"

#input_shape = 10*10*512   #this is the shape of bottleneck feature of each image which comes after passing the image through VGG-16
input_shape = 512
model = model(input_shape)
# model.load_weights(os.path.join(SAVER, "model.h5"))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])

epochs = 10
batch_size = 10
step = 0
Train_bottleneck_files = int(len(train) / batch_size)
CV_bottleneck_files = int(len(cv) / batch_size)

epoch_number, Train_loss, Train_acc, CV_loss, CV_acc = [], [], [], [], []

for epoch in range(epochs):
    avg_epoch_Tr_loss, avg_epoch_Tr_acc, avg_epoch_CV_loss, avg_epoch_CV_acc = 0, 0, 0, 0
    epoch_number.append(epoch + 1)
    
    for i in range(Train_bottleneck_files):
        
        step += 1
        
        #loading batch of train bottleneck features for training MLP.
        X_Train_load = np.load(os.path.join(SAVEDIR_TRAIN, "bottleneck_{}.npy".format(i+1)))
        X_Train = X_Train_load.reshape(X_Train_load.shape[0], X_Train_load.shape[1]*X_Train_load.shape[2]*X_Train_load.shape[3])
        Y_Train = np.load(os.path.join(SAVEDIR_TRAIN_LABELS, "bottleneck_labels_{}.npy".format(i+1)))
        #print(X_Train.shape[1])
        
        #loading batch of Human CV bottleneck features for cross-validation.
        X_CV_load = np.load(os.path.join(SAVEDIR_CV, "bottleneck_{}.npy".format((i % CV_bottleneck_files) + 1)))
        X_CV = X_CV_load.reshape(X_CV_load.shape[0], X_CV_load.shape[1]*X_CV_load.shape[2]*X_CV_load.shape[3])
        Y_CV = np.load(os.path.join(SAVEDIR_CV_LABELS, "bottleneck_labels_{}.npy".format((i % CV_bottleneck_files) + 1)))
        
        #print("b4")
        Train_Loss, Train_Accuracy = model.train_on_batch(X_Train, Y_Train) #train the model on batch
        #print(Train_loss)
        CV_Loss, CV_Accuracy = model.test_on_batch(X_CV, Y_CV) #cross validate the model on CV Human batch
        #print("aft4")
        
        print("Epoch: {}, Step: {}, Tr_Loss: {}, Tr_Acc: {}, CV_Loss: {}, CV_Acc: {}".format(epoch+1, step, np.round(float(Train_Loss), 2), np.round(float(Train_Accuracy), 2), np.round(float(CV_Loss), 2), np.round(float(CV_Accuracy), 2)) )
        
        avg_epoch_Tr_loss += Train_Loss / Train_bottleneck_files
        #print(avg_epoch_Tr_loss)
        avg_epoch_Tr_acc += Train_Accuracy / Train_bottleneck_files
        avg_epoch_CV_loss += CV_Loss / Train_bottleneck_files
        avg_epoch_CV_acc += CV_Accuracy / Train_bottleneck_files
      
        
    print("Avg_Train_loss: {}, Avg_CombTrain_Acc: {}, Avg_CV_Loss: {}, Avg_CV_Acc: {}".format(np.round(float(avg_epoch_Tr_loss), 2), np.round(float(avg_epoch_Tr_acc), 2), np.round(float(avg_epoch_CV_loss), 2), np.round(float(avg_epoch_CV_acc), 2)))

    
    
    Train_loss.append(avg_epoch_Tr_loss)

    Train_acc.append(avg_epoch_Tr_acc)
    CV_loss.append(avg_epoch_CV_loss)
    CV_acc.append(avg_epoch_CV_acc)
    
    model.save(os.path.join(SAVER, "model.h5"))  #saving the model on each epoc
    model.save_weights(os.path.join(SAVER, "model_weights.h5")) #saving the weights of model on each epoch
    #print("Model and weights saved at epoch {}".format(epoch + 1))
          
log_frame = pd.DataFrame(columns = ["Epoch", "Train_Loss", "Train_Accuracy", "CV_Loss", "CV_Accuracy"])
log_frame["Epoch"] = epoch_number
log_frame["Train_Loss"] = Train_loss
log_frame["Train_Accuracy"] = Train_acc
log_frame["CV_Loss"] = CV_loss
log_frame["CV_Accuracy"] = CV_acc

log_frame.to_csv("F:/deepak/Thesis_test_code/ann/Logs/Log.csv", index = False)


log = pd.read_csv("F:/deepak/Thesis_test_code/ann/Logs/Log.csv")
print(log)