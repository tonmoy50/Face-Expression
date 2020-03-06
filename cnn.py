import keras,os
from keras.preprocessing import image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np





def get_model(height, width):
    #vgg16 model of Convolutional network
    model = Sequential()

    layer1 = 10
    layer2 = 15
    layer3 = 20
    layer4 = 25
    layer5 = 30

    model.add( Conv2D( input_shape=(height, width, 3), filters=layer1, kernel_size=(3,3), padding="same", activation="relu" ) )
    model.add( Conv2D( filters=layer1, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add( Conv2D( filters=layer2, kernel_size=(3,3), padding="same", activation="relu" ) )
    #model.add( Dropout(0.5) )
    model.add( Conv2D( filters=layer2, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )
    model.add( Dropout(0.5) )
    model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )
    model.add( Conv2D( filters=layer3, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )
    model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )
    model.add( Dropout(0.5) )
    model.add( Conv2D( filters=layer4, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )
    model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )
    model.add( Dropout(0.5) )
    model.add( Conv2D( filters=layer5, kernel_size=(3,3), padding="same", activation="relu" ) )

    model.add( MaxPool2D( strides=(2,2), pool_size=(2,2) ) )

    model.add(Flatten())

    model.add( Dense(100, activation="relu") )
    #model.add( Dropout(0.5) )
    model.add( Dense(100, activation="relu") )

    model.add( Dense(7, activation="softmax") )

    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model
    




def main():
    height = 128
    width = 128
    train_obj = ImageDataGenerator()
    train_data = train_obj.flow_from_directory(directory="img_tr", target_size=(width, height))
    test_obj = ImageDataGenerator()
    test_data = test_obj.flow_from_directory(directory="img_test", target_size=(width,height))


    model = get_model(height, width)
    #print(train_data)
    #model.summary()

    check_point = ModelCheckpoint("cnn_model.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='acc', min_delta=20, verbose=1, mode='auto', patience=10)

    khela = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data, validation_steps=10, epochs=100 , callbacks=[check_point, early_stop])
    
    print("Done!!!!!!!!!!!!!!!!!!!!!")



#if __name__ == "__main__":
main()