#!/usr/bin/python

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dropout,MaxPool2D,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from math import ceil
import os
import numpy as np
import glob
import cv2
import random

import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']="-1" #to use only the cpu


class Model:

	def shuffleDivideData(self,data,labels):
		#we shuffle the data
		generator_state=np.random.get_state()
		np.random.shuffle(data)
		np.random.set_state(generator_state)
		np.random.shuffle(labels)
		
		#normalise the data
		#we divide the data into training set and test set
		self.trainx,x,self.trainy,y=train_test_split(data,labels,test_size=0.20)
		self.testx,self.valx,self.testy,self.valy=train_test_split(x,y,test_size=0.50)
		
		'''
		self.testx=self.testx/255.0
		self.valx=self.valx/255.0
		'''
		#don't preprocess first
				
		
		print("feature shape: ",self.trainx.shape[1:])
		print("Training Samples: ",len(self.trainy))
		print("Validation Samples: ",len(self.valx))
		print("Test Samples: ",len(self.testy))
	
	def readClasses(self):
		print("reading the classes")
		self.classes=np.array(os.listdir("Dataset"))
		
		self.no_output=len(self.classes) #no of the output classes
		print("number of classes: ",self.no_output)
		print("Classes: ",self.classes)
		np.savez_compressed("classes",self.classes)
		
		
		data=[]
		labels=[]
		for class_no,class_name in enumerate(self.classes):
			print("reading class ",class_name)
			images_path=glob.glob("Dataset/"+class_name+"/*.jpg")
			random.shuffle(images_path)
			for image_path in images_path:
				image=cv2.imread(image_path)
				image=cv2.resize(image,(200,200))
				
				data.append(image)
				labels.append(class_no)
		
		print("finished reading")
		
		self.shuffleDivideData(np.array(data),np.array(labels)) #to get the test dataset
		
		np.savez_compressed("trainingset",self.trainx,self.trainy)
		np.savez_compressed("validationset",self.valx,self.valy)
		np.savez_compressed("testset",self.testx,self.testy)
		print("saved the training,validation and test set")
		
		'''	
		self.image_generator=ImageDataGenerator(rescale=1./255,zoom_range=0.45,horizontal_flip=True,
				brightness_range=[0.2,1.8],rotation_range=10)
		
		self.train_data_generator=self.image_generator.flow(self.trainx,self.trainy,batch_size=self.batch_size,shuffle=True)
		
		print("data generator created")
		
		#the number of steps of each epoch
		self.epoch_steps=ceil(self.trainx.shape[0]/self.batch_size)
		'''
		
	
	def __init__(self):
		
		self.batch_size=32
		self.readClasses()
		'''
		#we define the model
		self.defineModel()
		
		print(self.model.summary())
		#we compile the model
		self.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
		
		self.history=self.model.fit(self.train_data_generator,steps_per_epoch=self.epoch_steps,epochs=50,validation_data=(self.valx,self.valy))
		self.model.evaluate(self.testx,self.testy)
		'''
		
		
	
	def defineModel(self):
		
		self.model=Sequential()
		
		self.model.add(Conv2D(32,3,activation='relu',input_shape=self.trainx.shape[1:]))
		#self.model.add(Dropout(0.5))
		self.model.add(MaxPool2D((5,5),strides=1,padding='same'))
		self.model.add(Conv2D(32,3,activation='relu'))
		self.model.add(Dropout(0.35))
		self.model.add(Conv2D(16,3,activation='relu'))
		self.model.add(MaxPool2D((3,3),strides=1,padding='same'))
		self.model.add(Dropout(0.35))
		self.model.add(Flatten())
		self.model.add(Dense(128,kernel_initializer='he_normal',activation='relu'))
		self.model.add(Dropout(0.25))
		self.model.add(Dense(64,kernel_initializer='he_normal',activation='relu'))
		self.model.add(Dropout(0.25))
		self.model.add(Dense(self.no_output,kernel_initializer='he_normal',activation='softmax'))
	
	def plotCurves(self):
		
		#learning curve
		plt.subplot(2,1,1)
		plt.title("Learning Curve")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.plot(self.history.history['loss'],label='train')
		plt.plot(self.history.history['val_loss'],label='validation')
		plt.legend()
		
		plt.subplot(2,1,2)
		plt.title("Accuracy")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.plot(self.history.history['accuracy'],label='train')
		plt.plot(self.history.history['val_accuracy'],label="validation")
		plt.legend()
		plt.show()
	
	def saveEverything(self):
		
		#we save the trained model
		self.model.save("Model/SimpleClassifier")
		print("model saved successfully")
		
		#we save the classes
		np.savez_compressed("Model/classes",self.classes)
		
		print("saving complete")
		
		


classifier=Model()
#classifier.saveEverything()
#classifier.plotCurves()
