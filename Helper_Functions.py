#!/usr/bin/python
import re
import pandas as pd
import scipy
import numpy as np

import Import_Files

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
    il_ctm,mo_ctm = pd.read_csv(Import_Files.il_trim_list), pd.read_csv(Import_Files.mo_trim_list)
    il_ctm,mo_ctm = il_ctm[il_ctm['WORKTYPE'] == 'Cycle Maintenance'],mo_ctm[mo_ctm['WORKTYPE'] == 'Cycle Maintenance']
    #collapse cycle maintainance trims that cross over from december to january
    # over different years
    circuit_id_ctm_list = []
    mo_ctm.DUEDATE , il_ctm.DUEDATE = mo_ctm.DUEDATE.astype(str).str.slice(0,4),il_ctm.DUEDATE.astype(str).str.slice(0,4)
    #Clean the sub-feeder   - zero filling
    for i in  il_ctm.iterrows():
        try:
            if i[1]['LINENAME'].find('-') == -1:
                if(len(i[1]['LINENAME'])== 6):
                    circuit_id_ctm_list.append((i[1]['LINENAME'][0:3].zfill(3)) +'-' +(i[1]['LINENAME'][3:6].zfill(3)) + '-' + i[1]['DUEDATE'])
            else:
                circuit_id_ctm_list.append((str.split(i[1]['LINENAME'],'-')[0].zfill(3)) +'-' +(str.split(i[1]['LINENAME'],'-')[1].zfill(3)) + '-' + i[1]['DUEDATE'])
        except:
            pass
    for i in  mo_ctm.iterrows():
        try:
            if i[1]['LINENAME'].find('-') == -1:
                if(len(i[1]['LINENAME'])== 6):
                    circuit_id_ctm_list.append((i[1]['LINENAME'][0:3].zfill(3)) +'-' +(i[1]['LINENAME'][3:6].zfill(3)) + '-' + i[1]['DUEDATE'])
            else:
                circuit_id_ctm_list.append((str.split(i[1]['LINENAME'],'-')[0].zfill(3)) +'-' +(str.split(i[1]['LINENAME'],'-')[1].zfill(3)) + '-' + i[1]['DUEDATE'])
        except:
            pass
    ## Extrapolate missing schedule records
    sub_fed_dict = {}
    for i in circuit_id_ctm_list:
        sub,fed,year = i.split('-')
        if (sub + '-' + fed) in sub_fed_dict.keys():
            sub_fed_dict[sub + '-' + fed].append(year)
        else:
            sub_fed_dict[sub + '-' + fed] = [year]
    # Extrapolate N times
    N = 2
    for sub_fed in sub_fed_dict.keys():
        try:
            sub_fed_dict[sub_fed] = sorted(list(set([int(x) for x in sub_fed_dict[sub_fed]])))
            mode = scipy.stats.mode(np.diff(sub_fed_dict[sub_fed])).mode[0]
            for i in range(0,N):
                new_year = (min(sub_fed_dict[sub_fed])-mode)
                sub_fed_dict[sub_fed].append(new_year)
                circuit_id_ctm_list.append(sub_fed + '-' + str(new_year))
        except:
            pass
    return(circuit_id_ctm_list)

def substationFeederStandardizer(substation_feeder):
    if len(substation_feeder) > 8:
        return np.NaN
    try:
        sub,feeder = str.split(substation_feeder,'-')
        # If the substation name format is SSS-F
        if len(feeder) == 1:
            feeder = '00' + feeder
            substation_feeder = sub + '-' + feeder
             # If the substation name format is SSS-FF
        if len(feeder) == 2:
            feeder = '0' + feeder
            substation_feeder = sub + '-' + feeder
   # If the substation name format is SSSFFF
    except:
        sub = substation_feeder[0:3]
        feeder = substation_feeder[3:6]
        substation_feeder = sub + '-' + feeder
    return(substation_feeder)

def standardizeCrewType(crew_type):
    dusty_crewtype = {"planner":"jobplanner","manual":"climbing","climbing_t":"climbing","climbing":"climbing",
                      "mower":"mowing","mowing_t":"mowing","70":"bucket","070":"bucket",
                      "Back Yard Lift":"byl","mackenzie":"other","abo":"other","gcl":"other",
                      "2":"other","02":"other","wr_781":"other","wr_1251":"other","jarraff":"jarraff",
                      "helicopter":"helicopter","tractor":"tractor",'bucket':'bucket','jobplanner':'jobplanner'}
    crew_type = str(crew_type).lower()
    crew_type = re.sub(r'\W+', '_', crew_type)
    if crew_type in dusty_crewtype:
        return dusty_crewtype[crew_type]
    else:
        return 'other'
        
    
def returnNextTrim(circuit_id,ctm_cycles):
    similar_trims = [x for x in ctm_cycles if x.startswith(circuit_id)]
    
    
    
    
    
    
    