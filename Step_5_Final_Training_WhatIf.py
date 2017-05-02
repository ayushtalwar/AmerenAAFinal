#!/usr/bin/python
import pandas as pd
import numpy as np
import os
import Import_Models
import Import_Files
import Step_3_PipelineModel
import Step_2_CrewDataFactory
import Step_2_CrewTrimHistoryFactory
import Helper_Functions
import Reliability_Helper_Functions
import pickle
from sklearn.externals import joblib
import datetime
pd.options.mode.chained_assignment = None  # default='warn'
with open(Import_Files.useful_circuits,'rb') as f:
    useful_circuits_list = pickle.load(f)
with open(Import_Files.total_crew_per,'rb') as f:
    total_crew_per = pickle.load(f)
current_yr = datetime.datetime.now().year
# Load Data From disk
cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data = Step_3_PipelineModel.loadMasterData()
# Create Training Data
master_df = Step_3_PipelineModel.joinData(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data)
master_df = master_df.dropna()
# Filter out 2017 trims
master_df ['yr'] = [x.split('-')[2] for x in master_df.circuit_id]
master_df = master_df[master_df.yr.astype(int) < current_yr]
master_df['curr_total_units'] = (master_df.curr_trims_per_mi +master_df.curr_rem_per_mi + master_df.curr_brush_per_mi)  * master_df.curr_cust_per_mi
master_df['curr_avg_crew_week_bckt'] = [total_crew_per[x]['bucket_per'] for x in master_df.circuit_id]
master_df['curr_avg_crew_week_clmbng'] = [total_crew_per[x]['climbing_per'] for x in master_df.circuit_id]
master_df['curr_avg_crew_week_othr'] = [total_crew_per[x]['others_per'] for x in master_df.circuit_id]             
train_dat = master_df
target_feat = ['curr_total_units','curr_trims_per_mi','curr_rem_per_mi','curr_brush_per_mi',
               'curr_total_hr','curr_trim_hr','curr_rem_hr',
                'curr_brush_hr','curr_bucket','curr_climbing',
                'curr_other','curr_avg_crew_week_bckt','curr_avg_crew_week_clmbng',
                'curr_avg_crew_week_othr','curr_total_cost']
def impute(df):
    df = df.replace([-np.nan,np.nan,np.inf,-np.inf],0)
    return(df)    
def createNoisyInput(df, feature):
    feature_mod = []
    for row in df.iterrows():
        var = row[1][feature]      
        if var != 0 :
            var_mod_ul = var + (var*0.2)
            var_mod_ll = var - (var * 0.2)
            var_scope = round(var/10) + 1
            var_mod = var
            while var_mod >= var_mod_ll and var_mod <= var_mod_ul:
                noise = np.random.normal(0,var_scope,1)
                var_mod = var + noise
            feature_mod.append(var_mod[0])
        else:
            var_mod_ul = 2
            var_mod_ll = - 2
            var_scope = 2
            var_mod = var
            while var_mod >= var_mod_ll and var_mod <= var_mod_ul:
                noise = np.random.normal(0,var_scope,1)
                var_mod = var + noise
            feature_mod.append(var_mod[0])           
    df = df.drop(feature,1)        
    df[feature] = feature_mod
    return(df)        
for t in target_feat:
    print(t)
    drop_feat_list = [col for col in train_dat.columns if 'curr' in col]
    drop_feat_list += ['circuit_id','prev_trim_yr','yrs_since_trim', 'prev_mean_unit', 'src_op_center','yr']
    drop_feat_list.remove('curr_trims_per_mi')
    drop_feat_list.remove('curr_rem_per_mi')
    y_train_unscaled = train_dat[t]
    X_train_unscaled = train_dat.drop(drop_feat_list,1)
    X_train_unscaled = X_train_unscaled.rename(columns = {'curr_trims_per_mi':'expert_trims',
                                                          'curr_rem_per_mi':'expert_rem'})
    X_train_unscaled = createNoisyInput(X_train_unscaled,'expert_trims')
    X_train_unscaled = createNoisyInput(X_train_unscaled,'expert_rem')    
    X_train_unscaled = impute(X_train_unscaled)
    y_train_unscaled = impute(y_train_unscaled)
    rf_grid_ = Step_3_PipelineModel.buildRandomForestRegCV(X_train_unscaled,y_train_unscaled)
    joblib.dump(rf_grid_, eval("Import_Models." + t + "_whatif_model")) 
  

