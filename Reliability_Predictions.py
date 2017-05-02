import os
import Import_Files
import Step_3_PipelineModel
import pandas as pd
import Step_4_SimilarityEngine
import Import_Models
from sklearn.externals import joblib
import Reliability_Functions
import datetime
import csv
import Reliability_Helper_Functions
pd.options.mode.chained_assignment = None  # default='warn'

# Load Data From disk
cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data = Step_3_PipelineModel.loadMasterData()
master_df = Step_3_PipelineModel.joinData(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data)

current_yr = int(datetime.datetime.now().year)
# Create  Testing 
sub_yr_dict = {}
testing_circuit_id = []

for circuit in master_df.circuit_id:
    subfed = circuit.split('-')[0] + '-' + circuit.split('-')[1]
    yr = circuit.split('-')[2]
    if subfed in sub_yr_dict.keys():
        sub_yr_dict[subfed].append(yr)
    else:
        sub_yr_dict[subfed] = [yr]

# Don't include current year trims 
for circuit,yr in sub_yr_dict.items():
    yr = [int(x) for x in yr if int(x) != current_yr]
    if len(yr) > 0:
        max_yr = max(yr)
        testing_circuit_id.append(circuit + '-' + str(max_yr))

test_dat = master_df[master_df['circuit_id'].isin(testing_circuit_id)]
class_model =  joblib.load(Import_Models.reliability_classification_model)
X_train = pd.read_pickle(Import_Files.reliability_train_data)
reg_model = joblib.load(Import_Models.reliability_regression_model)


prediction_final_df,prediction_final_cont_vlue = Reliability_Functions.testAndPredict(test_dat,X_train,class_model,reg_model) 
prediction_final_df.circuit_id = [x.split('-')[0] + '-' + x.split('-')[1] for x in prediction_final_df.circuit_id]
prediction_final_df = prediction_final_df.rename(columns = {'circuit_id':'CIRCUIT_NM',
                                                     'months_since_trim':'MONTHS_SNCE_LST_TRIM_CNT',
                                                     'variable':'OUTAGE_CLS_NBR',
                                                     'value':'FAILURE_LIKELIHOOD_PCT'}) 

clss_rnge = {'0':'0-1','1':'2-3','2':'4-5'
             ,'3':'5-7','4':'8-9','5':'10-11','6':'12-24'}
prediction_final_df.OUTAGE_CLS_NBR = prediction_final_df.OUTAGE_CLS_NBR.astype(str)
prediction_final_df.OUTAGE_CLS_NBR = [clss_rnge[x] for x in prediction_final_df.OUTAGE_CLS_NBR]

sql_flg = 0
with open(Import_Files.sql_flag) as f:
    csv_f = csv.reader(f)
    for row in csv_f:
        if row[0] == '1':
            sql_flg = 1
if sql_flg == 1:
    prediction_final_df.to_sql('TVM_O_CIRCUIT_FAILURE', Import_Files.engine, if_exists='replace')
else:    
    prediction_final_df.to_csv(Import_Files.path_to_data +  "TVM_O_CIRCUIT_FAILURE.csv",index = False)
  
##########################################################################
trim_mode,all_ctm_cycles = Reliability_Helper_Functions.getCrewTrimMainCycles()
prediction_final_cont_vlue = prediction_final_cont_vlue.rename(columns = {prediction_final_cont_vlue.columns[0] :"No_of_otgs"})

percentile_df = pd.DataFrame()
df = prediction_final_cont_vlue

cyc_freq = []
for row in df.iterrows():
    circuit = row[1]['circuit_id'].split('-')[0] + '-' +row[1]['circuit_id'].split('-')[1] 
    if circuit in trim_mode.keys():
        cyc_freq.append(trim_mode[circuit])
    else:
        cyc_freq.append(4)

df['CYC_FREQ'] = cyc_freq
df['CYC_FREQ'][df['CYC_FREQ']<4] = 4
df['CYC_FREQ'][df['CYC_FREQ']>4] = 6
    
df = df.drop('circuit_id',1)

df_4_cyc = df[df.CYC_FREQ == 4]
df_4_cyc = df_4_cyc.drop('CYC_FREQ',1)
df_25 = df_4_cyc.groupby(by = ['months_since_trim'],as_index= False).quantile(q =0.25)
df_25 = df_25[df_25.months_since_trim <= 48]
df_25['PER'] = '25_PCT'
df_75 = df_4_cyc.groupby(by = ['months_since_trim'],as_index= False).quantile(q =0.75)
df_75 = df_25[df_75.months_since_trim <= 48]
df_75['PER'] = '75_PCT'
percentile_4_yr = df_25.append(df_75)     
percentile_4_yr['CYC_FREQ']  = 4

df_6_cyc = df[df.CYC_FREQ == 6]
df_6_cyc = df_6_cyc.drop('CYC_FREQ',1)
df_25 = df_6_cyc.groupby(by = ['months_since_trim'],as_index= False).quantile(q =0.25)
df_25['PER'] = '25_PCT'
df_75 = df_6_cyc.groupby(by = ['months_since_trim'],as_index= False).quantile(q =0.75)
df_75['PER'] = '75_PCT'
percentile_6_yr = df_25.append(df_75)     
percentile_6_yr['CYC_FREQ'] = 6
final_percentile_table = percentile_4_yr.append(percentile_6_yr)

final_percentile_table = final_percentile_table.rename(columns = {'No_of_otgs':'FAILURE_LIKELIHOOD_PCT',
                                                'months_since_trim':'MONTHS_SNCE_LST_TRIM_CNT'}) 

if sql_flg == 1:
    final_percentile_table.to_sql('TVM_O_CIRCUIT_FAILURE_PCT', Import_Files.engine, if_exists='replace')
else:    
    final_percentile_table.to_csv(Import_Files.path_to_data +  "TVM_O_CIRCUIT_FAILURE_PCT.csv",index = False)


    











