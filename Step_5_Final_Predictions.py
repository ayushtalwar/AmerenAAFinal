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
from sklearn.externals import joblib
import csv
with open(Import_Files.useful_circuits,'rb') as f:
    useful_circuits_list = pickle.load(f)
with open(Import_Files.total_crew_per,'rb') as f:
    total_crew_per = pickle.load(f)
# Re Run Factories to write files to disk
pd.options.mode.chained_assignment = None  # default='warn'
# Load Data From disk
cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data = Step_3_PipelineModel.loadMasterData()
# Create Training Data
master_df = Step_3_PipelineModel.joinData(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data)
master_df = master_df.dropna()
# Filter out 2017 trims
master_df ['yr'] = [x.split('-')[2] for x in master_df.circuit_id]
master_df = master_df[master_df.yr.astype(int) < 2017]
master_df['curr_total_units'] = (master_df.curr_trims_per_mi +master_df.curr_rem_per_mi + master_df.curr_brush_per_mi)  * master_df.curr_cust_per_mi
master_df['curr_avg_crew_week_bckt'] = [ total_crew_per[x]['bucket_per'] for x in master_df.circuit_id]
master_df['curr_avg_crew_week_clmbng'] = [total_crew_per[x]['climbing_per'] for x in master_df.circuit_id]
master_df['curr_avg_crew_week_othr'] = [total_crew_per[x]['others_per'] for x in master_df.circuit_id]             
train_dat = master_df
# Join Data Sources
trim_mode, all_trims = Reliability_Helper_Functions.getCrewTrimMainCycles()
master_df = Step_3_PipelineModel.joinData(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data)
cyc_freq = []
for row in master_df.iterrows():
    circuit = row[1]['circuit_id'].split('-')[0] + '-' + row[1]['circuit_id'].split('-')[1]
    if circuit in trim_mode.keys():
        cyc_freq.append(trim_mode[circuit])
    else:
        cyc_freq.append(4)
master_df['cyc_freq'] = cyc_freq
master_df['yr'] = [x.split('-')[2] for x in master_df.circuit_id]
master_df['nxt_trim_yr'] = master_df['yr'].astype(int) + master_df['cyc_freq']
current_yr = datetime.datetime.now().year
master_df = master_df[master_df['nxt_trim_yr'] >= current_yr] 
test_dat = master_df
test_dat['circuit_id'] = [x.split('-')[0] + '-' + x.split('-')[1] for x in test_dat.circuit_id]
test_dat['circuit_id'] = test_dat['circuit_id'] + '-' + test_dat['nxt_trim_yr'].astype(str)
test_dat.to_pickle(Import_Files.test_dat_disk)
final_prediction_df = test_dat[['circuit_id','curr_cust_per_mi']]
def impute(df):
    df = df.replace([-np.nan,np.nan,np.inf,-np.inf],0)
    return(df)
test_dat = impute(test_dat) 
# Select columns
columns_select = [col for col in test_dat.columns if 'curr' in col]
columns_select += ['customers','oh_mile','ug_mile','voltage',
                   'Line phases','Canopy cover','Horizontal','Other','Vertical',
                   'otgs_snce_lst_trim','cmi_snce_lst_trim']
# Change columns names on test data             
test_dat = test_dat[columns_select] 
test_dat = test_dat.drop("curr_cust_per_mi",1)
for c in test_dat.columns:
    if 'curr' in c:
        new_column_name = c.replace('curr','prev')
        test_dat = test_dat.rename(columns = {c:new_column_name})
target_feat = ['curr_total_units','curr_trims_per_mi','curr_rem_per_mi','curr_brush_per_mi',
               'curr_total_hr','curr_trim_hr','curr_rem_hr',
                'curr_brush_hr','curr_bucket','curr_climbing',
                'curr_other','curr_avg_crew_week_bckt','curr_avg_crew_week_clmbng',
                'curr_avg_crew_week_othr','curr_total_cost']
for t in target_feat:
    model = joblib.load(eval("Import_Models." + t + '_model'))
    pred = model.predict(test_dat)
    final_prediction_df[t] = pred
## Create TVO_O_WORK_CIRCUIT                        
train_data_work_pred = train_dat[final_prediction_df.columns]                     
train_data_work_pred['Prediction_Indicator'] =0
final_prediction_df['Prediction_Indicator'] = 1
final_df = pd.concat([final_prediction_df,train_data_work_pred],0)                       
final_df['year'] = [x.split("-")[2] for x in final_df.circuit_id]                       
final_df.head()
final_df['curr_trims_per_mi'] = final_df['curr_trims_per_mi'] * final_df['curr_cust_per_mi']
final_df['curr_rem_per_mi'] = final_df['curr_rem_per_mi'] * final_df['curr_cust_per_mi']
final_df['curr_brush_per_mi'] = final_df['curr_brush_per_mi'] * final_df['curr_cust_per_mi']
final_df = final_df.rename(columns = {"curr_trims_per_mi":"no_units_trims",
                                      "curr_rems_per_mi":"no_units_rems",
                                      "curr_brush_per_mi":"no_units_brush"})                 
final_df = final_df.drop('curr_cust_per_mi',1)
contractor_dict = {}
for key,value in crew_obj.crew_data_dict.items():
    contractor = value[0]['contractor']
    contractor_dict[key] = contractor
contractors = []
for row in final_df.iterrows():
    if row[1]['circuit_id'] in contractor_dict.keys():
        contractors.append(contractor_dict[row[1]['circuit_id']])
    else:
        contractors.append("Unavailable")
final_df['contractors'] = contractors
final_df['circuit_id'] = [x.split("-")[0] + "-" + x.split("-")[1] for x in final_df.circuit_id]
final_df = final_df.rename(columns = {'circuit_id':'CIRCUIT_NM','year':'MTNS_YR',
                                      'Prediction_Indicator':'PREDICTION_IND',
                                      'curr_total_units':'ALL_UNIT_CNT',
                                      'no_units_trims':'TRIM_UNIT_CNT',
                                      'curr_rem_per_mi':'REMOVAL_UNIT_CNT',
                                      'no_units_brush':'BRUSH_UNIT_CNT',
                                      'curr_total_hr':'ALL_MAN_HRS',
                                      'curr_trim_hr':'TRIM_MAN_HRS',
                                      'curr_rem_hr':'REMOVAL_MAN_HRS',
                                      'curr_brush_hr':'BRUSH_MAN_HRS',
                                      'curr_bucket':'BUCKET_CREW_HRS',
                                      'curr_climbing':'MANUAL_CREW_HRS',
                                      'curr_other':'OTHER_CREW_HRS',
                                      'curr_avg_crew_week_bckt':'BUCKET_CREW_WK_AVG',
                                      'curr_avg_crew_week_clmbng':'MANUAL_CREW_WK_AVG',
                                      'curr_avg_crew_week_othr':'OTHER_CREW_WK_AVG',
                                      'curr_total_cost':'TOTAL_COST_AMT',
                                      'contractors':'CONTRACTOR_NM'})
# Normalize crew mix hours by pushing excess/under hours in climbing
final_df['t'] = final_df.BUCKET_CREW_HRS +   final_df.MANUAL_CREW_HRS + final_df.OTHER_CREW_HRS                     
final_df['t'] = 1 - final_df['t']
final_df.MANUAL_CREW_HRS  = final_df.MANUAL_CREW_HRS + final_df.t  
final_df.MANUAL_CREW_HRS = final_df.MANUAL_CREW_HRS * final_df.ALL_MAN_HRS
final_df.BUCKET_CREW_HRS = final_df.BUCKET_CREW_HRS * final_df.ALL_MAN_HRS
final_df.OTHER_CREW_HRS = final_df.OTHER_CREW_HRS * final_df.ALL_MAN_HRS
final_df = final_df.drop('t',1)
sql_flg = 0
with open(Import_Files.sql_flag) as f:
    csv_f = csv.reader(f)
    for row in csv_f:
        if row[0] == '1':
            sql_flg = 1
if sql_flg == 1:
    final_df.to_sql('TVM_O_CIRCUIT_WORK_DATA', Import_Files.engine, if_exists='replace')
else:    
    final_df.to_csv(Import_Files.path_to_data +  "TVM_O_CIRCUIT_WORK_DATA.csv",index = False)
########################################################################
# Create TVM_O_CIRCUIT
trim_mode,ctm_list = Reliability_Helper_Functions.getCrewTrimMainCycles()
cth_object = Step_2_CrewTrimHistoryFactory.CrewTrimHistoryFactory(Import_Files.total_cleaned_cth_data_input,
                                                         Import_Files.cth_factory_file_f,
                                                         Import_Files.carryover_f,Import_Files.last_trim_month_collector)
final_prediction_df['prev_total_cost'] = test_dat['prev_total_cost']
final_prediction_df['otgs_snce_lst_trim'] = test_dat['otgs_snce_lst_trim']
final_prediction_df['Canopy cover'] = test_dat['Canopy cover']
final_df = pd.DataFrame()
final_df = pd.concat([final_prediction_df],0)                                      
predicted_df  = pd.DataFrame()
predicted_df['CIRCUIT_NM'] = final_df.circuit_id
predicted_df['PRED_COST_AMT'] = final_df.curr_total_cost
predicted_df['PRED_MAN_HRS'] = final_df.curr_total_hr
predicted_df['YEAR'] = [x.split("-")[2] for x in predicted_df['CIRCUIT_NM']]
predicted_df['sub_fed'] = [x.split("-")[0] + '-' + x.split("-")[1] for x in predicted_df.CIRCUIT_NM]
CYCLE_FREQUENCY_CNT = []
for i in predicted_df.sub_fed:
    if i in trim_mode.keys():
        CYCLE_FREQUENCY_CNT.append(trim_mode[i])
    else:
        CYCLE_FREQUENCY_CNT.append(4)
predicted_df['CYCLE_FREQUENCY_CNT']  = CYCLE_FREQUENCY_CNT
predicted_df['NXT_MNTC_YR'] = predicted_df['CYCLE_FREQUENCY_CNT'] + predicted_df['YEAR'].astype(int)
predicted_df['LAST_CYCLE_COST_AMT'] = final_df.prev_total_cost
predicted_df['OTGS_SNCE_LAST_TRIM_CNT'] = final_df.otgs_snce_lst_trim
predicted_df['VEGETATION_PCT'] = final_df['Canopy cover']
predicted_df['CIRCUIT_NM'] = [x.split("-")[0] + "-" + x.split("-")[1] for x in predicted_df.CIRCUIT_NM]
predicted_df_dict = {}
for row in predicted_df.iterrows():
    if row[1]['CIRCUIT_NM'] in predicted_df_dict.keys():
        predicted_df_dict[row[1]['CIRCUIT_NM']].append(row[1]['YEAR'])
    else:
        predicted_df_dict[row[1]['CIRCUIT_NM']] = [row[1]['YEAR']]                                                  
# De dup table
dups = []
for k,v in predicted_df_dict.items():
    if len(v) > 1:
        dups.append(k)
blacklist_circuits = []
for circuit in dups:
    min_x = min(predicted_df_dict[circuit])
    for v in predicted_df_dict[circuit]:
        if int(v) > int(min_x):
            blacklist_circuits.append(circuit + "-" + v)
predicted_df['circuit_id'] = predicted_df.CIRCUIT_NM + "-" + predicted_df.YEAR
predicted_df['keep'] = 1       
keep = [x for x in predicted_df.circuit_id if x not in blacklist_circuits]            
keep = pd.DataFrame(keep,columns = ['circuit_id'])
predicted_df = predicted_df.merge(keep,on = 'circuit_id', how= 'inner')
predicted_df = predicted_df.drop(['keep','sub_fed','circuit_id'],1) 
if sql_flg == 1:
    predicted_df.to_sql('TVM_O_CIRCUIT', Import_Files.engine, if_exists='replace')
else:    
    predicted_df.to_csv(Import_Files.path_to_data + "TVM_O_CIRCUIT.csv",index = False)
                       