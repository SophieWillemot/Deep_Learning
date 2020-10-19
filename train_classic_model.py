import SimpleITK as sitk 
import numpy as np 
import matplotlib.pyplot as plt 
import json 
import csv 
import os 
import pandas as pd 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 

from classification.pre_process.Prep_CSV import Prep_CSV
from classification.pre_process.Preprocessing import Preprocessing 
from classification.model.classi_model import *
from utils.modality_CT import *


json_path = '/media/deeplearning/78ca2911-9e9f-4f78-b80a-848024b95f92/result.json'
nifti_directory = '/media/deeplearning/78ca2911-9e9f-4f78-b80a-848024b95f92'
objet = Prep_CSV(json_path)
objet.result_csv(nifti_directory)
print(objet.csv_result_path)

prep_objet = Preprocessing(objet.csv_result_path)
X, y = prep_objet.normalize_encoding_dataset()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42, test_size = 0.15) #random state 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 42, test_size = 0.15)
print("size of X_train : ", X_train.shape)
print("size of y_train : ",y_train.shape)
print("")
print("size of X_test : ", X_test.shape)
print("size of y_test : ",y_test.shape)
print("")
print("size of X_val : ", X_val.shape)
print("size of y_val : ",y_val.shape)

plt.imshow(X_train[0])
plt.show()

model = classic_model(input_shape=(503, 136, 1))
model.summary()