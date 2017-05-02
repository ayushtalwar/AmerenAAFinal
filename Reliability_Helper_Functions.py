#!/usr/bin/python
import pandas as pd
import scipy
import numpy as np
import Import_Files
import datetime 
from scipy import stats
import dateutil
from dateutil.relativedelta import relativedelta
import re
import  Step_2_CrewTrimHistoryFactory
import Helper_Functions
import pickle
def clean_circuit_id(raw_id):
        if raw_id == np.nan or len(raw_id) > 6 or re.search('[^A-Za-z0-9]', raw_id):
            return '000-000'
        elif len(raw_id) < 6:
            raw_id = raw_id.zfill(6)
        return raw_id[:3] + "-" + raw_id[3:]
        
    
def getPrevTrim(circuit_id, history):
        sub, fed, year = circuit_id.split('-')
        # Find previous trim keys  for same circuit
        feeder_previous_trims = [key for key in history.keys() if key.startswith((sub + '-' + fed))]
        feeder_years = []
        for i in feeder_previous_trims:
            if (int(i.split("-")[2])) < int(year):
                feeder_years.append(int(i.split("-")[2]))
        try:
            prev_trim_year = max(feeder_years)
            prev_circuit_id = sub + "-" + fed + "-" + str(prev_trim_year)
            return(prev_circuit_id ,prev_trim_year, int(year))
        except:
            return(0, np.NaN, int(year))

def getCrewTrimMainCycles():
    
    circuit_id_ctm_list = list(pd.read_pickle(Import_Files.cth_factory_file_f).circuit_id)
    with open(Import_Files.last_trim_month_collector,'rb') as f:
        last_trim_month_dict = pickle.load(f) 
    orig_sched = Helper_Functions.getCrewTrimMainCycles()
    circuit_list  = [x.split('-')[0] + '-' + x.split('-')[1] for x in circuit_id_ctm_list]
    circuit_list = list(set(circuit_list))
    circuit_mode_dict = {}
    final_circuit_id_ctm_list = []
    # Get modes from orig sched 
    for circuit in circuit_list:
        try:
            circuit_trim_yrs = np.sort([int(x.split('-')[2]) for x in orig_sched if x.startswith(circuit)])
            yrs_bw_trims = np.ediff1d(circuit_trim_yrs)
            mode = scipy.stats.mode(yrs_bw_trims).mode[0]
            if mode < 4:
                mode = 4
            if mode == 5 or mode == 8 or mode == 7:
                mode = 6
            circuit_mode_dict[(circuit.split("-")[0] + '-' + circuit.split("-")[1])] = mode
        except:
            pass                      
    missing_circuit_modes = [x for x in circuit_list if x not in circuit_mode_dict.keys()]
    missing_circuit_modes = list(set(missing_circuit_modes))
    
    for circuit in missing_circuit_modes:
        circuit_trim_yrs = np.sort([int(x.split('-')[2]) for x in circuit_id_ctm_list if x.startswith(circuit)])
        if len(circuit_trim_yrs)> 1:
            yrs_bw_trim = list(np.ediff1d(circuit_trim_yrs))
            mode =  scipy.stats.mode(yrs_bw_trim).mode[0]
            circuit_mode_dict[circuit] = mode
        if len(circuit_trim_yrs)  == 0:
            circuit_mode_dict[circuit] = 4
                     
    ## Extrapolate missing schedule records
    sub_fed_dict = {}
    all_dates = []
    for circuit_trim in circuit_id_ctm_list:
        sub,fed,year = circuit_trim.split('-')
        date = last_trim_month_dict[circuit_trim]
        if (sub + '-' + fed) in sub_fed_dict.keys():
            sub_fed_dict[sub + '-' + fed].append(datetime.datetime.strptime((year + '-' + str(date)),"%Y-%m"))
        else:
            sub_fed_dict[sub + '-' + fed] = [datetime.datetime.strptime((year + '-' + str(date)),"%Y-%m")]
        # Extrapolate N times
    N = 4
    for sub_fed in sub_fed_dict.keys():
        try:
            mode = circuit_mode_dict[sub_fed]
            for i in range(0,N):
                new_year = min(sub_fed_dict[sub_fed]) - relativedelta(years=(i * mode))
                all_dates.append(new_year)
                final_circuit_id_ctm_list.append(sub_fed + '-' + datetime.datetime.strftime(new_year,"%Y-%m"))
                new_year = max(sub_fed_dict[sub_fed]) + relativedelta(years=(i * mode))
                final_circuit_id_ctm_list.append(sub_fed + '-' + datetime.datetime.strftime(new_year,"%Y-%m"))
        except:
            pass
    return(circuit_mode_dict,final_circuit_id_ctm_list)