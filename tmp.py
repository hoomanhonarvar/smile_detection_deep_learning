import keras
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from keras import datasets, layers, models
import joblib


#val_ds=tf.keras.utils.image_dataset_from_directory(
 #   'files',
  #  validation_split=0.2,
   # subset="validation",
    #seed=123,
    #image_size=(64,64),
   # batch_size=2
#)
catagoris=['not_Smile','Smile']
X=[]
num_train=3000
Y=[]
data_dir='./files/'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


for i in catagoris:
    print(f'loading ... catagoris:{i}')
    print(i)
    path=os.path.join(data_dir,i)
    x = 0


    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img))
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        if len(faces) > 0:
            for j, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 255), 2)
                face = img_array[y:y + h, x:x + w]
            face=cv2.resize(face,(64,64))
            # cv2.imshow("",face)
            # cv2.waitKey(0)
            X.append(face)
            if i=="Smile":
                Y.append(1)
            else:
                Y.append(0)
            x+=1
    if x==num_train:
        break

    print(f'loaded category:{i} successfully')
    print(len(X))
X=np.array(X)
Y=np.array(Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=2)
x_train,validx,y_train,validy=train_test_split(x_train,y_train,test_size=0.3,random_state=2)

x_train, x_test = x_train / 255.0, x_test / 255.0
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1,activation="sigmoid"))

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs=15

history=model.fit(
    x_train,y_train,
    batch_size=10,
    validation_data=(validx, validy),
    epochs=epochs
)


test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(test_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation L oss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
joblib.dump(model,'traind_model.joblib')