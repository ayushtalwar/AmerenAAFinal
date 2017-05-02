#!/usr/bin/python
import pandas as pd
import numpy as np
import os
import Import_Files
import Step_3_PipelineModel
import Step_2_CrewDataFactory
import Step_2_CrewTrimHistoryFactory
import Helper_Functions
import Reliability_Helper_Functions
import pickle
import Import_Models
import datetime
import csv
import sys
from sklearn.externals import joblib
import json
# system arguments inputs
external_trims=   sys.argv[1] 
external_rem=  sys.argv[2]
circuit_id=  sys.argv[3]
year =   sys.argv[4]
output_file  = sys.argv[5]
external_trims = float(external_trims)
external_rem = float(external_rem)
# read test dat from disk
test_dat = pd.read_pickle(Import_Files.test_dat_disk)
def impute(df):
    df = df.replace([-np.nan,np.nan,np.inf,-np.inf],0)
    return(df)
# mod
circuit_trim = str(circuit_id) + '-' + str(year)
test_dat = impute(test_dat) 
test_dat = test_dat[test_dat.circuit_id == circuit_trim ]
# Select columns
columns_select = [col for col in test_dat.columns if 'curr' in col]
columns_select += ['customers','oh_mile','ug_mile','voltage',
                   'Line phases','Canopy cover','Horizontal','Other','Vertical',
                   'otgs_snce_lst_trim','cmi_snce_lst_trim']
# Change columns names on test data             
test_dat = test_dat[columns_select] 
cust_pr_mile = test_dat.curr_cust_per_mi
test_dat = test_dat.drop("curr_cust_per_mi",1)
test_dat['external_trims'] = external_trims / cust_pr_mile
test_dat['external_rem'] = external_rem / cust_pr_mile
for c in test_dat.columns:
    if 'curr' in c:
        new_column_name = c.replace('curr','prev')
        test_dat = test_dat.rename(columns = {c:new_column_name})
target_feat = ['curr_total_units','curr_trims_per_mi','curr_rem_per_mi','curr_brush_per_mi','curr_total_hr','curr_trim_hr','curr_rem_hr',
                'curr_brush_hr','curr_bucket','curr_climbing','curr_other','curr_avg_crew_week_bckt','curr_avg_crew_week_clmbng',
                'curr_avg_crew_week_othr','curr_total_cost']
final_pred_dict = {}
for t in target_feat:
    model = joblib.load(eval("Import_Models." + t + '_whatif_model'))
    pred = model.predict(test_dat)
    final_pred_dict[t] = pred[0]   
final_pred_dict ['BUCKET_CREW_WK_AVG_CNT'] = final_pred_dict.pop('curr_avg_crew_week_bckt')
final_pred_dict ['ALL_UNIT_CNT'] = final_pred_dict.pop('curr_total_units')
final_pred_dict ['TRIM_UNIT_CNT'] = final_pred_dict.pop('curr_trims_per_mi')
final_pred_dict ['REMOVAL_UNIT_CNT'] = final_pred_dict.pop('curr_rem_per_mi')
final_pred_dict ['BRUSH_UNIT_CNT'] = final_pred_dict.pop('curr_brush_per_mi')
final_pred_dict ['ALL_MAN_HRS'] = final_pred_dict.pop('curr_total_hr')
final_pred_dict ['TRIM_MAN_HRS'] = final_pred_dict.pop('curr_trim_hr')
final_pred_dict ['REMOVAL_MAN_HRS'] = final_pred_dict.pop('curr_rem_hr')
final_pred_dict ['BRUSH_MAN_HRS'] = final_pred_dict.pop('curr_brush_hr')
final_pred_dict ['BUCKET_CREW_HRS'] = final_pred_dict.pop('curr_bucket')
final_pred_dict ['MANUAL_CREW_HRS'] = final_pred_dict.pop('curr_climbing')
final_pred_dict ['OTHER_CREW_HRS'] = final_pred_dict.pop('curr_other')
final_pred_dict ['MANUAL_CREW_WK_AVG_CNT'] = final_pred_dict.pop('curr_avg_crew_week_clmbng')
final_pred_dict ['OTHER_CREW_WK_AVG_CNT'] = final_pred_dict.pop('curr_avg_crew_week_othr')
final_pred_dict ['TOTAL_COST_AMT'] = final_pred_dict.pop('curr_total_cost')
# Normalize crew mix hours
t = final_pred_dict['BUCKET_CREW_HRS'] + final_pred_dict['MANUAL_CREW_HRS'] + final_pred_dict['OTHER_CREW_HRS']
diff = 1 - t
final_pred_dict['MANUAL_CREW_HRS'] = final_pred_dict['MANUAL_CREW_HRS'] + diff
final_pred_dict['MANUAL_CREW_HRS'] = final_pred_dict['MANUAL_CREW_HRS'] * final_pred_dict ['ALL_MAN_HRS']
final_pred_dict['BUCKET_CREW_HRS'] = final_pred_dict['BUCKET_CREW_HRS'] * final_pred_dict ['ALL_MAN_HRS']
final_pred_dict['OTHER_CREW_HRS'] = final_pred_dict['OTHER_CREW_HRS'] * final_pred_dict ['ALL_MAN_HRS']
# Convert trim and removal predictions to absolute numbers 
final_pred_dict ['TRIM_UNIT_CNT'] = final_pred_dict ['TRIM_UNIT_CNT'] * int(cust_pr_mile)
final_pred_dict ['REMOVAL_UNIT_CNT'] = final_pred_dict ['REMOVAL_UNIT_CNT'] * int(cust_pr_mile)               
#Output
with open(output_file,'w') as f:
    json.dump(final_pred_dict, f)








    
    
    
