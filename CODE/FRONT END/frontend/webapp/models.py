from django.db import models
from keras.models import load_model
import cv2
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json
from PIL import Image


from tensorflow.keras.models import load_model
cnn=load_model(r'C:\Users\Hemanth\Music\Cotton Diesease\CODE\FRONT END\frontend\model_cnn.h5')
vgg16=load_model(r'C:\Users\Hemanth\Music\Cotton Diesease\CODE\FRONT END\frontend\model_vgg16.h5')

def predict(img,algo): 
	file = Image.open(img)
	img = file.resize((224,224))
	img_array = np.asarray(img).astype(np.uint8)
	res = img_array.reshape([-1,224, 224,3])
	print(res.shape)
	if algo=='vgg16':
		y_pred=np.argmax(vgg16.predict(res),axis=1)
		return y_pred[0]
	else:
		y_pred=np.argmax(cnn.predict(res),axis=1)
		return y_pred[0]



