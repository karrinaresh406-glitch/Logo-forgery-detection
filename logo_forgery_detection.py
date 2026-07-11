from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import pickle
import os
from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

main = tkinter.Tk()
main.title("Fake Logo Detection")
main.geometry("1300x1200")

global filename
global classifier
global labels, X, Y, X_train, y_train, X_test, y_test

def readLabels(filename):
    global labels
    labels = []
    for root, dirs, files in os.walk(filename):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)

def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    readLabels(filename)
    text.insert(END,"Logo found in dataset are\n\n")
    for i in range(len(labels)):
        text.insert(END,labels[i]+"\n")

def processDataset():
    text.delete('1.0', END)
    global filename, X, Y, X_train, y_train, X_test, y_test, labels

    X = []   # FIX 1
    Y = []   # FIX 1

    for root, dirs, files in os.walk(filename):
        for file in files:   # FIX 2
            if file.endswith(".jpg") or file.endswith(".png"):
                path = os.path.join(root, file)
                img = cv2.imread(path)

                if img is not None:
                    img = cv2.resize(img,(64,64))
                    im2arr = np.array(img)
                    X.append(im2arr)

                    name = os.path.basename(root)
                    label = labels.index(name)
                    Y.append(label)

    X = np.asarray(X)
    Y = np.asarray(Y)

    X = X.astype('float32')
    X = X/255

    text.insert(END,"Dataset Preprocessing Completed\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    Y = to_categorical(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    text.insert(END,"80% images are used to train CNN : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images are used to test CNN : "+str(X_test.shape[0])+"\n")

def trainCNN():
    text.delete('1.0', END)
    global classifier, X_train, y_train, X_test, y_test, labels, X, Y

    classifier = Sequential()
    classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Convolution2D(32,(3,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(256,activation='relu'))
    classifier.add(Dense(y_train.shape[1],activation='softmax'))

    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    classifier.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))

    predict = classifier.predict(X_test)
    predict = np.argmax(predict,axis=1)
    testY = np.argmax(y_test,axis=1)

    p = precision_score(testY,predict,average='macro')*100
    r = recall_score(testY,predict,average='macro')*100
    f = f1_score(testY,predict,average='macro')*100
    a = accuracy_score(testY,predict)*100

    text.insert(END,"CNN Accuracy  : "+str(a)+"\n")
    text.insert(END,"CNN Precision : "+str(p)+"\n")
    text.insert(END,"CNN Recall    : "+str(r)+"\n")
    text.insert(END,"CNN FSCORE    : "+str(f)+"\n\n")

    conf_matrix = confusion_matrix(testY,predict)

    plt.figure(figsize=(6,6))
    ax = sns.heatmap(conf_matrix,xticklabels=labels,yticklabels=labels,annot=True,cmap="viridis",fmt="g")
    plt.title("CNN Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def classifyLogo():
    global classifier, labels
    filename = filedialog.askopenfilename(initialdir="testImages")

    image = cv2.imread(filename)
    img = cv2.resize(image,(64,64))

    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)

    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255

    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img,(700,400))

    cv2.putText(img,'Logo Predicted as : '+labels[predict],(10,25),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.imshow('Result',img)
    cv2.waitKey(0)

def close():
    main.destroy()

font=('times',16,'bold')
title=Label(main,text='Fake Log Detection')
title.config(bg='LightGoldenrod1',fg='medium orchid')
title.config(font=font,height=3,width=120)
title.place(x=0,y=5)

font1=('times',13,'bold')

upload=Button(main,text="Upload Logo Dataset",command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)

pathlabel=Label(main)
pathlabel.config(bg='yellow4',fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=50,y=150)

processButton=Button(main,text="Preprocess Dataset",command=processDataset)
processButton.place(x=50,y=200)
processButton.config(font=font1)

trainButton=Button(main,text="Train CNN Algorithm",command=trainCNN)
trainButton.place(x=50,y=250)
trainButton.config(font=font1)

classifyButton=Button(main,text="Logo Classification",command=classifyLogo)
classifyButton.place(x=50,y=300)
classifyButton.config(font=font1)

exitButton=Button(main,text="Exit",command=close)
exitButton.place(x=50,y=350)
exitButton.config(font=font1)

font1=('times',12,'bold')

text=Text(main,height=25,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=370,y=100)
text.config(font=font1)

main.config(bg='burlywood2')
main.mainloop()
