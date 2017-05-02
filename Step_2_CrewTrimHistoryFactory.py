#!/usr/bin/python
import datetime
import pandas as pd
import csv
import numpy as np
import datetime
import Helper_Functions
import Import_Files
import pickle
import csv

def previousTrimAggregator(crew_trim_history_master):
    prev_trim_df = pd.DataFrame()
    output_df = pd.DataFrame()
    # Column names for final data frame
    column_names = ['date','no_qtr_spans',
                    'brush_acres','no_of_rem','no_of_trims',
                    'qsp_hr','brush_hr','rem_hr',
                     'trim_hr','total_cost', 'total_hr']
    for circuit_id in crew_trim_history_master.keys():
        # Get current trim data
        curr_data = crew_trim_history_master[circuit_id]
        curr_data_df = pd.DataFrame(curr_data, columns=column_names)
        curr_agg_dict = cthAggregateFunction(curr_data_df, prefix="curr")
        prev_circuit_id, prev_trim_yr, curr_trim_yr = Helper_Functions.getPrevTrim(circuit_id, crew_trim_history_master)
        # Get previous trim data
        if np.isnan(prev_trim_yr):
            # Store NaN vals if no prev trim exists
            prev_cols = ['prev_' + col for col in column_names if col != 'date']
            nan_vals = [np.NaN for col in column_names if col != 'date']
            prev_agg_dict = dict(zip(prev_cols, nan_vals))
        else:
            prev_data = crew_trim_history_master[prev_circuit_id]
            prev_data_df = pd.DataFrame(prev_data, columns=column_names)
            prev_agg_dict = cthAggregateFunction(prev_data_df, prefix="prev")
            prev_trim_df = prev_trim_df.append({'circuit_id': circuit_id, 'prev_circuit_id': prev_circuit_id}, ignore_index=True)
        output_dict = dict(curr_agg_dict)
        output_dict.update(prev_agg_dict)
        output_dict['prev_trim_yr'] = prev_trim_yr
        output_dict['yrs_since_trim'] = curr_trim_yr - prev_trim_yr
        output_dict['circuit_id'] = circuit_id
        output_df = output_df.append(output_dict, ignore_index = True)
    prev_trim_df.to_csv(Import_Files.prev_trim_f, index=False)
    return(output_df)

def featureStats (input_feature):
    input_feature = pd.to_numeric(input_feature)
    return(
           input_feature.sum(skipna = "True"),
           )

def cthAggregateFunction(previous_trim_data_df, prefix):
    final_agg_features = {}
    start_date = pd.to_datetime(previous_trim_data_df.date).min()
    end_date = pd.to_datetime(previous_trim_data_df.date).max()
    final_agg_features[prefix + '_' + 'durtn_days'] = (end_date - start_date).days
    n_col = previous_trim_data_df.shape[1]
    for i in previous_trim_data_df.columns[1:n_col]:
        feat_names = [prefix + '_' + i]#,"mean_" + i,"std_" + i,"skew_" + i]
        feat_values = featureStats(previous_trim_data_df[i])
        for j in range(0,len(feat_names)):
            final_agg_features[feat_names[j]] = feat_values[j]
    return(final_agg_features)


class CrewTrimHistoryFactory():
    def __init__(self, total_cleaned_cth_data_input,output_pickle_file, carryover_f, last_trim_month_collector):
        self.total_cleaned_cth_data_input = total_cleaned_cth_data_input
        self.df = pd.DataFrame()
        self.output_pickle_file = output_pickle_file
        self.crew_trim_history_master = {}
        self.crew_trim_history_master_relevant = {}
        self.crew_trim_history_master_recalibrated = {}
        self.carryover_f = carryover_f
        self.crew_trim_cycle_maintenance = list(set(Helper_Functions.getCrewTrimMainCycles()))
        self.circuit_id_month_lookup = {}
        self.last_trim_month_collector = last_trim_month_collector
        sql_flg = 0
        with open(Import_Files.sql_flag) as f:
            csv_f = csv.reader(f)
            for row in csv_f:
                if row[0] == '1':
                    sql_flg = 1
        with open( self.total_cleaned_cth_data_input) as f:
            csv_f = csv.reader(f)
            counter = 0
            for row in csv_f:
                if counter>0:
                    substation_feeder = str(row[0])
                    # Filter out substation names that we do not have legitimate id's for currently and OPEN circuits
                    if len(substation_feeder) < 9:
                        org_name = str(row[0])
                        substation_feeder = Helper_Functions.substationFeederStandardizer(substation_feeder)
                        year = str(row[2])[0:4]
                        # Select  data
                        total_hr = float(row[8]) + float(row[9]) + float(row[10]) + float(row[11])
                        data_sample = [row[2], row[4], row[5], row[6],
                                        row[7], row[8],
                                        row[9], row[10], row[11],
                                        row[12], total_hr]
                        circuit_id = substation_feeder + '-' + year
                        if circuit_id in self.crew_trim_history_master.keys():
                            self.crew_trim_history_master[circuit_id].append(data_sample)
                        else:
                            self.crew_trim_history_master[circuit_id] = [data_sample]
                counter = counter + 1
        self.crew_trim_all_netbs = self.crew_trim_history_master
        ## Last Trim Month
        self.month_collector = {}
        for circuit_id,data in self.crew_trim_history_master.items():
            circuit_dates = []
            for d in data:
                circuit_dates.append(datetime.datetime.strptime(d[0],"%Y%m%d").date())
            circuit_dates = min(circuit_dates)    
            self.month_collector[circuit_id] = circuit_dates.month    
        with open(self.last_trim_month_collector,'wb') as f:
            pickle.dump(self.month_collector, f, pickle.HIGHEST_PROTOCOL)

     
        ### Deal with cross over from Dec Y1 to Jan Y2
        blacklist_circuit_id = self.createBlackList()
        
        # Filter out blacklisted id's
        for circuit_id in self.crew_trim_history_master.keys():
            if circuit_id not in blacklist_circuit_id:
                self.crew_trim_history_master_recalibrated[circuit_id] = self.crew_trim_history_master[circuit_id]
            else:
                pass
        self.crew_trim_history_master = self.crew_trim_history_master_recalibrated
        
        ## Filter out non cycle trims as best as feasible 
        self.useful_circuits,self.dates,self.trims,self.costs = self.initialRecursion()
        self.useful_circuits_cp, self.non_cycle_trims_cp = self.recursionIteration2()
        self.keep_trims = {}
        
        ## Keep only useful _circuits 
        for circuit_id , data in self.crew_trim_history_master.items() :
            if circuit_id in list(set(self.useful_circuits_cp)):
                self.keep_trims [circuit_id] = data
            else:
                pass                    
        self.df  = previousTrimAggregator(self.keep_trims)
        
    def getFeatures(self):
        return(self.df)

    def writeFeatures(self):
        self.df.to_pickle(self.output_pickle_file)
        with open(Import_Files.useful_circuits,'wb') as f:
            pickle.dump(self.useful_circuits_cp, f, pickle.HIGHEST_PROTOCOL)
        
            
    def initialRecursion(self):
        circuit_ids = {}
        costs = {}
        trims = {}
        dates = {}
        for key,value in self.crew_trim_history_master.items():
            local_cost = 0
            local_dates = []
            local_trims = 0
            for sample in value:
                local_cost += float(sample[9])
                local_dates.append(sample[0])
                local_trims += float(sample[4])
            costs[key] = local_cost    
            dates[key] = local_dates
            trims[key] = local_trims     
    
        circuit_yr_dict = {}
        for key in self.crew_trim_history_master.keys():
            new_key = key.split("-")[0] + "-" + key.split("-")[1]
            yr = key.split("-")[2]
            if new_key in circuit_yr_dict.keys():  
                circuit_yr_dict[new_key].append( yr)
            else:
                circuit_yr_dict[new_key] = [yr]
        
        useful_circuits = []          
        for key,value in circuit_yr_dict.items():
            value = [int(x) for x in value]
            viable_pairs = {}
            for v in value:
                local_value = [x for x in value if x-v > 3]
                if len(local_value) > 0:
                    cost_v = costs[key + "-" + str(v)]
                    costs_x = [{x:cost_v + costs[key + "-" + str(x)]} for x in local_value]          
                    costs_x_mx = max([  cost_v + costs[key + "-" + str(x)]  for x in local_value]) 
                    for yr in costs_x:              
                        if list(yr.values())[0] == costs_x_mx:
                            mx_yr = list(yr.keys())[0]
                        viable_pairs[(str(v) + "_" + str(mx_yr))] = costs_x_mx
                if len(viable_pairs) > 0:
                    mx_viable_pair_cst = max([x for x in viable_pairs.values()])
                    final_mx_pair = 0
                    final_mx_cost = 0
                    for pair,pr_cost in viable_pairs.items():
                        if pr_cost == mx_viable_pair_cst:
                            final_mx_pair = pair
                            final_mx_cost = pr_cost
                    useful_circuits.append(key + "-" + final_mx_pair.split("_")[0])    
                    useful_circuits.append(key + "-" + final_mx_pair.split("_")[1])    
        return(useful_circuits,dates,trims,costs)

    
    def recursionIteration2(self):
        non_cycle_trims = [x for x in self.crew_trim_history_master.keys() if x not in self.useful_circuits]    
        non_cycle_trims_cp = non_cycle_trims.copy()
        ## From non cycle trims to useful trims
        ## Hand pick based on specific heuristics - flags
        for circuit in non_cycle_trims:          
            flg1  = 0
            flg2 = 0
            flg3 = 0
            feasible_trim_years = [int(circuit.split('-')[2]), (int(circuit.split('-')[2])-1), (int(circuit.split('-')[2])+1)]
            feasible_trims = [circuit.split('-')[0] + '-' + circuit.split('-')[1] + '-' + str(x) for x in feasible_trim_years]
            if feasible_trims[0] in self.crew_trim_cycle_maintenance or feasible_trims[1] in self.crew_trim_cycle_maintenance or feasible_trims[2] in self.crew_trim_cycle_maintenance:
                flg1 = 1
            if len(set(self.dates[circuit])) > 1:
                flg2 = 1
            if self.trims[circuit] > 0:
                flg3 = 1    
            if flg1 == 1 and flg2 == 1 and flg3 == 1:
                self.useful_circuits.append(circuit)
                non_cycle_trims_cp.remove(circuit)
            useful_circuits_cp = self.useful_circuits.copy()
        ## From  useful trims to non cycle trims 
        ## Hand pick based on specific heuristics - flags
        for circuit in self.useful_circuits:
            flg1 = 0
            flg2 = 0
            flg3 = 0
            if self.trims[circuit] == 0:
                flg1 = 1
            if len(set(self.dates[circuit])) ==1:
                flg2 = 1
            feasible_trim_years = [int(circuit.split('-')[2]), (int(circuit.split('-')[2])-1)]
            feasible_trims = [circuit.split('-')[0] + '-' + circuit.split('-')[1] + '-' + str(x) for x in feasible_trim_years]
            if circuit not in feasible_trims:
                flg3 = 1        
            if (flg1 == 1 and flg2 == 1) or flg3 == 1 :
                useful_circuits_cp.remove(circuit)
                non_cycle_trims_cp.append(circuit)
        return(useful_circuits_cp,non_cycle_trims_cp)       

    def createBlackList(self):
        blacklist_circuit_id = []
        for circuit_id in self.crew_trim_history_master.keys():
            # Check if there is a circuit id in the next year
            circuit_id_next_yr = circuit_id.split('-')[0] + '-' + circuit_id.split('-')[1] +'-' + str(int(circuit_id.split('-')[2]) +1)
            if circuit_id_next_yr in self.crew_trim_history_master.keys():
                circuit_id_dates = [(datetime.datetime.strptime(x[0],"%Y%m%d")).date() for x in self.crew_trim_history_master[circuit_id]]
                circuit_id_next_year_dates = [(datetime.datetime.strptime(x[0],"%Y%m%d")).date() for x in self.crew_trim_history_master[circuit_id_next_yr]]
                circuit_id_max = max(circuit_id_dates)
                circuit_id_next_year_dates_min = min(circuit_id_next_year_dates)
                # If the trim in the next year is within 14 days of the trim in the previous year
                difference_between_dates = (circuit_id_next_year_dates_min - circuit_id_max)
                if difference_between_dates < datetime.timedelta(days = 14):
                    temp_dict = self.crew_trim_history_master[circuit_id]
                    blacklist_circuit_id.append(circuit_id)
                    ## Change the year  to the following year
                    for crew_data_sample in range(0,(len(temp_dict))):
                        temp_dict[crew_data_sample][0] = str(circuit_id_next_year_dates_min).replace("-","")
                        self.crew_trim_history_master[circuit_id_next_yr].append(temp_dict[crew_data_sample])
        pd.Series(blacklist_circuit_id, name='circuit_id').to_csv(self.carryover_f, header=True, index=False)      
        return(blacklist_circuit_id)