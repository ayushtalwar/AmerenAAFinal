#!/usr/bin/python
import os
import Import_Files
import Step_2_CrewTrimHistoryFactory
import Reliability_Helper_Functions
import csv
import re
import numpy as np
import datetime
from datetime import timedelta
import Step_2_WeatherFactory
from geopy.distance import vincenty
import pandas as pd
import Helper_Functions
import Step_3_PipelineModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import Step_2_WeatherFactory
pd.options.mode.chained_assignment = None  # default='warn'

# Create list of extended trims
# Create a dictionary with key = substation_feeder and 
# Values are trim dates

def createAllTrimsDict():
    trim_mode,extended_trim_cycles = Reliability_Helper_Functions.getCrewTrimMainCycles()
    all_trims_dict = {}
    for circuit_trim in extended_trim_cycles:
        sub,fed,year,date = str.split(circuit_trim,'-')
        if (sub + '-' + fed) in all_trims_dict.keys():
            all_trims_dict[sub + '-' + fed].append(year + '-' + date)
        else:
            all_trims_dict[sub + '-' + fed] = [year + '-' + date]
    return(extended_trim_cycles,all_trims_dict)        
 
    

# Appropriate each outage to its corresponding trim from the extended trims dict
def createOutageTrimsDF(all_trims_dict):
    final_outage_df = []       
    with open(Import_Files.outage_data_f) as f:
        csv_f = csv.reader(f)
        counter = 0
        for row in csv_f:
            if counter >0 and row[10] == 'EXTENDED' and int(row[7]) > 0 :
                    sub_fed = Reliability_Helper_Functions.clean_circuit_id(row[1])
                    if sub_fed in all_trims_dict.keys():
                        outage_dt = datetime.datetime.strptime(str(row[8][0:9]),'%d%b%Y').date()
                        outage_feeder_trim_dates = all_trims_dict[sub_fed]
                        closest_trim_date = 0
                        closest_trim_days = 99999
                        for trim_date in outage_feeder_trim_dates:
                            days_from_outage = outage_dt - datetime.datetime.strptime(trim_date,"%Y-%m").date()
                            if days_from_outage > datetime.timedelta(days = 0) and days_from_outage.total_seconds()/(24*3600) < closest_trim_days:
                                closest_trim_date = trim_date
                                closest_trim_days = days_from_outage
                                closest_trim_days = closest_trim_days.total_seconds()/(24*3600)
                        final_outage_df.append((sub_fed,outage_dt,closest_trim_date, closest_trim_days))  
            counter += 1       
    return(final_outage_df)



#Create Reliability Weather DF
# Output no of events per weather station and no of days where the
# Max wind gust speed is higher than mean + 2 std per station
# This takes into account all the data that the station has and is meant to
# qualitatively assert how prone to gusts and events is a substation
# There is no time dependency of the events or max gust days
def createReliabilityWeather():
    
    weather_dat = pd.read_pickle(Import_Files.total_weather_data)
    weather_dat = weather_dat[['Weather_Stn',' Events',
                           ' Max Wind SpeedMPH',' Max Gust SpeedMPH']]    
    imp_events = ['Fog-Rain-Hail-Thunderstorm','Fog-Rain-Snow-Thunderstorm',
              'Fog-Rain-Thunderstorm','Fog-Rain-Thunderstorm-Tornado',
              'Fog-Snow-Thunderstorm','Fog-Thunderstorm','Rain-Hail-Thunderstorm',
              'Rain-Hail-Thunderstorm-Tornado','Rain-Snow-Hail-Thunderstorm','Rain-Snow-Thunderstorm',
              'Rain-Thunderstorm','Rain-Thunderstorm-Tornado','Snow-Thunderstorm','Thunderstorm']
    weather_dat_dict = {}
    for weather_stn in list(set(weather_dat.Weather_Stn)):
        stn_dat = weather_dat[weather_dat.Weather_Stn == weather_stn]
        max_gust_no_days  = stn_dat[stn_dat[' Max Gust SpeedMPH'] > (weather_dat[' Max Gust SpeedMPH'].mean() + (2*weather_dat[' Max Gust SpeedMPH'].std()))].shape[0]
        no_events = stn_dat[stn_dat[' Events'].isin(imp_events)].shape[0]
        weather_dat_dict[weather_stn] = (max_gust_no_days,no_events)
    return(weather_dat_dict)
#trim_mode,extended_trim_cycles = Reliability_Helper_Functions.getCrewTrimMainCycles()


# Create DataFrame with cumulative outages for each trim
# Return weather events from substation - weatherstn lookup 
def reliabilityDataFrame():
    extended_trim_cycles,all_trims_dict = createAllTrimsDict()
    final_outage_df = createOutageTrimsDF(all_trims_dict)
    final_rel_df = pd.DataFrame(final_outage_df,index = None,columns = ['sub_fed','outage_dt','closest_trim','days_to_outage']) 
    final_rel_df = final_rel_df[final_rel_df.closest_trim != 0].drop_duplicates()
    final_rel_df['year'] = [int(str(x).split('-')[0]) for x in final_rel_df.outage_dt]
    
    # Include outage after 2014
    final_rel_df = final_rel_df[final_rel_df.year >= 2014]
    
    # Drop outages that point to a circuit that is from the current year 
    trim_mode, all_trims =  Reliability_Helper_Functions.getCrewTrimMainCycles()
    all_trims = [x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2] for x in all_trims]
    current_year = str(datetime.datetime.now().year)
    trims_schd_current_yr = [x for x in all_trims if x.endswith(current_year)]
    trims_schd_current_yr = [x.split('-')[0] + '-' + x.split('-')[1] for x in trims_schd_current_yr]
    final_rel_df['keep'] = 1
    final_rel_df['keep'][(final_rel_df.sub_fed.isin(trims_schd_current_yr)) & (final_rel_df.year == int(current_year))] = 0
    final_rel_df = final_rel_df[final_rel_df.keep == 1] 
    final_rel_df = final_rel_df.drop('keep',1)


    all_otg_dict = {}
    for i in final_rel_df.iterrows():
        sud_fed_date = i[1]['sub_fed'] +"-" + i[1]['closest_trim']
        key = i[1]['sub_fed'] + '-' + i[1]['closest_trim'].split('-')[0] + "-" + i[1]['closest_trim'].split('-')[1]
        if key in all_otg_dict.keys():
            all_otg_dict[key].append(round(i[1]['days_to_outage']/30))
        else:
            all_otg_dict[key] = [(round(i[1]['days_to_outage']/30))]
    # Find cumulative trims
    final_list_otg = []
    for circuit_id,months_from_trim in all_otg_dict.items(): 
        try:
            closest_trim_months = 12 * trim_mode[(circuit_id.split('-')[0] + '-' + circuit_id.split('-')[1])]
            for i in range(1,closest_trim_months,1):
                months_from_trim = list(set(months_from_trim))
                months_from_trim_i = len([x for x in months_from_trim if x <= i])
                sample = (circuit_id,i,months_from_trim_i)
                final_list_otg.append(sample)
        except:
            pass
        
        
    final_df = pd.DataFrame(final_list_otg, columns = ['circuit_id','months_since_trim','no_of_otg'])
    '''
    # Add weather data
    sub_weather_lookup = Step_2_WeatherFactory.WeatherFactory(Import_Files.cth_factory_file_f, Import_Files.total_weather_data,
                      Import_Files.weather_factory_f,
                     Import_Files.weather_coordinates_f, Import_Files.ss_coordinates_f).sub_weather_lookup
    final_df['sub'] = [x.split('-')[0] for x in final_df.circuit_id]
    final_df = final_df[final_df['sub'].isin(sub_weather_lookup.keys())]
    final_df['wthr_stn'] = [sub_weather_lookup[x] for x in final_df['sub']]
    
    weather_dat_dict = createReliabilityWeather()
    final_df['max_gust_days'] = [weather_dat_dict[x][0] for x in final_df['wthr_stn']] 
    '''
    return(final_df)
    
def impute( df):
    df = df.replace([-np.inf,np.inf,np.nan],0)
    return df    
    

def relFactory():
    # Reliability data frame
    df = reliabilityDataFrame()
    df['circuit_id'] = [x.split("-")[0] + "-" + x.split("-")[1] + "-" + x.split("-")[2] for x in df['circuit_id']]

    # Master data frame - subset to current columns only
    cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data = Step_3_PipelineModel.loadMasterData()
    master_df = Step_3_PipelineModel.joinData(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data)
    curr_cols = [col for col in master_df.columns if 'curr' in col]
    curr_cols += ['circuit_id','oh_mile','Canopy cover','otgs_snce_lst_trim']
    master_df = master_df[curr_cols]
    # Merge the 2 data frames
    df = df.merge(master_df,on = 'circuit_id',how = 'inner')
    df['sub_fed'] = [x.split('-')[0] + '-' +x.split('-')[1] for x in df['circuit_id']] 
    return(df)

def sampleAndTrain(train_cid,df):
    train_dat = df[df.circuit_id.isin(train_cid)]
 #   train_dat = train_dat.drop(['sub','wthr_stn','circuit_id','sub_fed'],1)
    train_dat = train_dat.drop([ 'circuit_id','sub_fed'],1)
    bins = [0,2,4,6,8,10,25]
    y_train = train_dat['no_of_otg']    
    train_dat['binned_y'] = np.digitize(y_train, bins = bins)    
    
    ## Modify training data - stratified sampling
    train_dat_1 = train_dat[train_dat['binned_y'] == 1]
    train_dat_2 = train_dat[train_dat['binned_y'] == 2]
    train_dat_3 = train_dat[train_dat['binned_y'] == 3]    
    train_dat_4 = train_dat[train_dat['binned_y'] == 4]    
    train_dat_others =  train_dat[train_dat['binned_y'] >4]    
    train_dat_1_rs,test_1 = train_test_split(train_dat_1,train_size = 0.2)
    train_dat_2_rs,test_2 = train_test_split(train_dat_2,train_size = 0.6)
    train_dat_3_rs,test_3 = train_test_split(train_dat_3,train_size = 0.6)    
    train_dat_4_rs,test_4 = train_test_split(train_dat_4,train_size = 0.6)    
    train_dat = pd.DataFrame()
    train_dat = train_dat.append(train_dat_1_rs,ignore_index = True)
    train_dat = train_dat.append(train_dat_2_rs,ignore_index = True)
    train_dat = train_dat.append(train_dat_3_rs,ignore_index = True)
    train_dat = train_dat.append(train_dat_4_rs,ignore_index = True)
    train_dat = train_dat.append(train_dat_others,ignore_index = True)
    
    y_train = train_dat['binned_y']
    X_train = train_dat
    X_train = X_train.drop(['no_of_otg','binned_y'],1)
    X_train = impute(X_train)
    y_train = y_train.astype(str)
    model = RandomForestClassifier(n_estimators = 1500, max_depth = 21,
                              max_features = 4,min_samples_split = 3,
                              min_samples_leaf = 3)
    model.fit(X_train,y_train)
    return(model,train_dat,X_train,y_train)

def sampleAndTrainRegression(train_cid,df):
    train_dat = df[df.circuit_id.isin(train_cid)]
 #   train_dat = train_dat.drop(['sub','wthr_stn','circuit_id','sub_fed'],1)
    train_dat = train_dat.drop([ 'circuit_id','sub_fed'],1)
    y_train = train_dat['no_of_otg']    
    X_train = train_dat
    X_train = X_train.drop(['no_of_otg'],1)
    X_train = impute(X_train)
    model = RandomForestRegressor(n_estimators = 1500, max_depth = 21,
                              max_features = 4,min_samples_split = 3,
                              min_samples_leaf = 3)
    model.fit(X_train,y_train)
    return(model)

def testAndPredict(test_dat,X_train,class_model,reg_model):
    prediction_final_df = pd.DataFrame()
    prediction_final_class = []
    prediction_final_cont_vlue = pd.DataFrame()
    ctr = 0
    for row in test_dat.iterrows():
        ctr += 1
        if ctr %100 == 0:
            print (ctr)
        df = test_dat.loc[np.repeat(row[0],72)]
        df['months_since_trim'] = [x for x in range(1,73)]
        df = df[X_train.columns]
        df = impute(df)
        pred_prob = class_model.predict_proba(df)
        pred_cont_vlue = reg_model.predict(df)
        pred_df = pd.DataFrame(pred_prob)
        c = [x for x in pred_df.columns]
        pred_df['circuit_id'] = row[1]['circuit_id']
        pred_df['months_since_trim'] = [x for x in range(1,73)]
        pred_df_melt = pd.melt(pred_df,id_vars = ['circuit_id','months_since_trim'],
                               value_vars = c)
        prediction_final_df = prediction_final_df.append(pred_df_melt)       
        pred_cont_vlue_df = pd.DataFrame(pred_cont_vlue)
        pred_cont_vlue_df['circuit_id'] = row[1]['circuit_id']
        pred_cont_vlue_df['months_since_trim'] = [x for x in range(1,73)]
        prediction_final_cont_vlue = prediction_final_cont_vlue.append(pred_cont_vlue_df)
    return(prediction_final_df,prediction_final_cont_vlue)
















