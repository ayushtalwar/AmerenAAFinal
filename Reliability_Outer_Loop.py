#!/usr/bin/python
import os
import pickle
import Reliability_Functions
import Step_3_PipelineModel
import Step_2_WeatherFactory
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy
import sklearn
import pandas as pd
import Import_Files
from sklearn.externals import joblib
import Import_Models
# Run factory
total_df = Reliability_Functions.relFactory()
total_df.to_pickle(Import_Files.path_to_total_rel_df)

# Read data
total_df = pd.read_pickle(Import_Files.path_to_total_rel_df)
train_cid = list(set(total_df.circuit_id))
total_df = total_df[total_df['circuit_id'].isin(train_cid)]

# Train model and save to disk
rf_model,train_dat,X_train,y_train = Reliability_Functions.sampleAndTrain(train_cid,total_df)
joblib.dump(rf_model, Import_Models.reliability_classification_model) 
X_train.to_pickle(Import_Files.reliability_train_data)

rf_model_reg = Reliability_Functions.sampleAndTrainRegression(train_cid,total_df)
joblib.dump(rf_model_reg, Import_Models.reliability_regression_model) 

 

