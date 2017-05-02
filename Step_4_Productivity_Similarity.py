#!/usr/bin/python
import pandas as pd
import Step_3_PipelineModel
import Step_4_SimilarityEngine
import datetime
import csv
import Import_Files
# Load Data From disk
cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data = Step_3_PipelineModel.loadMasterData()
# Join Data Sources
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
#trims_select = master_df['circuit_id'][master_df['year']== year]
trims_select = master_df[master_df['circuit_id'].isin(testing_circuit_id)]['circuit_id']
trim_list = ['circuit_id','curr_cust_per_mi', 'curr_trims_per_mi',
			    'curr_trim_hr_per_mi',
			   'curr_tot_hr_per_mi', 'curr_bucket', 'curr_climbing', 'curr_helicopter',
			   'curr_jarraff', 'otgs_snce_lst_trim', 'Canopy cover', 'oh_mile',
			   'yrs_since_trim']
rem_list = ['circuit_id','curr_cust_per_mi','curr_rem_per_mi',
			   'curr_rem_hr_per_mi',
			   'curr_tot_hr_per_mi', 'curr_bucket', 'curr_climbing', 'curr_helicopter',
			   'curr_jarraff', 'otgs_snce_lst_trim', 'Canopy cover', 'oh_mile',
			   'yrs_since_trim']
brush_list = ['circuit_id','curr_cust_per_mi',
			   'curr_brush_per_mi', 'curr_brush_hr_per_mi',
			   'curr_tot_hr_per_mi', 'curr_bucket', 'curr_climbing', 'curr_helicopter',
			   'curr_jarraff', 'otgs_snce_lst_trim', 'Canopy cover', 'oh_mile',
			   'yrs_since_trim', 'curr_trims_per_hr']

trim_engine = Step_4_SimilarityEngine.SimilarityEngine(trim_list, run_factories=False)
rem_engine = Step_4_SimilarityEngine.SimilarityEngine(rem_list, run_factories=False)
brush_engine = Step_4_SimilarityEngine.SimilarityEngine(brush_list, run_factories=False)

final_df = pd.DataFrame()
n = 100
for circuit in trims_select:
	circuit_df = pd.DataFrame()
	circuit_df['Compare_Order'] = range(1, n + 1)
	circuit_df['Trim_Circuit_Name'] = trim_engine.getOpSimilar(circuit)[1:n+1]
	circuit_df['Removal_Circuit_Name'] = rem_engine.getOpSimilar(circuit)[1:n+1]
	circuit_df['Brush_Circuit_Name'] = brush_engine.getOpSimilar(circuit)[1:n+1]
	circuit_df['Circuit_Name'] = circuit

def split(feature,final_df):
    year = [x.split("-")[2] for x in final_df[feature]]
    sub_fed = [x.split("-")[0] + "-" + x.split("-")[1] for x in final_df[feature]]
    final_df[feature] = sub_fed
    final_df[(feature + "_year")] = year
    return(final_df)  

feature ='Trim_Circuit_Name'
final_df = split(feature,final_df)

feature ='Removal_Circuit_Name'
final_df = split(feature,final_df)

feature ='Brush_Circuit_Name'
final_df = split(feature,final_df)
  

final_df['yr'] = [x.split('-')[2] for x in final_df.Circuit_Name]  
sql_flg = 0
with open(Import_Files.sql_flag) as f:
    csv_f = csv.reader(f)
    for row in csv_f:
        if row[0] == '1':
            sql_flg = 1
if sql_flg == 1:
    final_df.to_sql('TVM_O_COMPARABLE_CIRCUIT', Import_Files.engine, if_exists='replace')
else:    
    final_df.to_csv(Import_Files.path_to_data +  "TVM_O_COMPARABLE_CIRCUIT.csv",index = False)
   
