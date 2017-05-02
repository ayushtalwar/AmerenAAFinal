#!/usr/bin/python
import datetime
import pandas as pd
import scipy
import csv, re
import numpy as np
import Helper_Functions
from collections import Counter
import Import_Files
import pickle
class CrewDataFactory():
    def modifyName(self,string):
        string = string.lower()
        string = re.sub(r'\W+', '_', string)
        return(string)

    def __init__(self,crew_data_f,crew_factory_file_path,useful_circuits):
        self.crew_data_f = crew_data_f
        self.total_equipment = []
        self.total_contractor = []
        self.total_crewtype = []
        self.total_worktype = []
        self.crew_factory_file_path = crew_factory_file_path
        self.crew_data_dict =  {}
        self.crew_data_master_relevant = {}
        self.crew_data_dict_recalibrated = {}
        self.final_output = pd.DataFrame()
        self.final_feat_dict = {}
        self.total_crew_per = {}
        self.useful_circuits = useful_circuits
        self.dusty_crewtype = {"planner":"jobplanner","manual":"climbing","climbing_t":"climbing","climbing":"climbing",
                               "mower":"mowing","mowing_t":"mowing","70":"bucket","070":"bucket",
                               "Back Yard Lift":"byl","mackenzie":"metro","abo":"other","gcl":"other",
                               "2":"other","02":"other","wr_781":"other","wr_1251":"other","jarraff":"jarraff",
                               "helicopter":"helicopter","tractor":"tractor",'bucket':'bucket','jobplanner':'jobplanner'}
        self.loadRaw()
        self.getAvgCrewWeek()    
    def loadRaw(self):
        counter = 0
        with open(self.crew_data_f) as f:
            csv_f = csv.reader(f)
            for row in csv_f:
                if counter > 0:
                    activity = str(row[8])
                    linename = str(row[0])
                    if len(linename) < 9 and activity == 'SOTT':
                        try:
                            circuit_id = linename.split('-')[0].zfill(3) + "-" + linename.split('-')[1].zfill(3) + "-" + row[5][0:4]
                            crew_data_sample = {"unit" :float(row[4]),
                                                "crewtype":self.dusty_crewtype[self.modifyName(str(row[9]))],
                                                "contractor" :self.modifyName(str(row[6])),
                                                "equipment" :self.modifyName(str(row[7])),
                                                "worktype" :self.modifyName(str(row[2])),
                                                "cost" : float(row[11]),
                                                "date":str(row[5])}
                            # Append to global lists of all possible crewtype, equipment, contractor , wktype
                            self.total_equipment.append(crew_data_sample['equipment'])
                            self.total_contractor.append(crew_data_sample['contractor'])
                            self.total_crewtype.append(crew_data_sample['crewtype'])
                            self.total_worktype.append(crew_data_sample['worktype'])
                            if circuit_id in self.crew_data_dict.keys():
                                self.crew_data_dict[circuit_id].append(crew_data_sample)
                            else:
                                self.crew_data_dict[circuit_id] = [crew_data_sample]
                        except:
                            pass
                counter += 1

        ## Collapse trims that cross over into the later year
        ## e.g If a trim starts in dec 2016 and ends in jan 2017
        ## we push all the data to 2017
        ### Deal with cross over from Dec Y1 to Jan Y2
        blacklist_circuit_id = []
        for circuit_id in self.crew_data_dict.keys():
            # Check if there is a circuit id in the next year
            circuit_id_next_yr = circuit_id.split('-')[0] + '-' + circuit_id.split('-')[1] +'-' + str(int(circuit_id.split('-')[2]) +1)
            if circuit_id_next_yr in self.crew_data_dict.keys():
                circuit_id_dates = [(datetime.datetime.strptime(x['date'],"%Y%m%d")).date() for x in self.crew_data_dict[circuit_id]]
                circuit_id_next_year_dates = [(datetime.datetime.strptime(x['date'],"%Y%m%d")).date() for x in self.crew_data_dict[circuit_id_next_yr]]
                circuit_id_max = max(circuit_id_dates)
                circuit_id_next_year_dates_min = min(circuit_id_next_year_dates)

                # If the trim in the next year is within 14 days of the trim in the previous year
                difference_between_dates = (circuit_id_next_year_dates_min - circuit_id_max)
                if difference_between_dates < datetime.timedelta(days = 14):
                    temp_dict = self.crew_data_dict[circuit_id]
                    blacklist_circuit_id.append(circuit_id)
                    #self.crew_trim_history_master.pop(circuit_id)
                    ## Change the year  to the following year
                    ## on the sample and append it to the next years data
                    for crew_data_sample in range(0,(len(temp_dict))):
                        temp_dict[crew_data_sample][0] = str(circuit_id_next_year_dates_min).replace("-","")
                        self.crew_data_dict[circuit_id_next_yr].append(temp_dict[crew_data_sample])

        # Filter out blacklisted id's
        for circuit_id in self.crew_data_dict.keys():
            if circuit_id not in blacklist_circuit_id:
                self.crew_data_dict_recalibrated[circuit_id] = self.crew_data_dict[circuit_id]
            else:
                pass
        self.crew_data_dict = {}
        self.crew_data_dict = self.crew_data_dict_recalibrated
        ## Convert to set
        self.total_equipment = list(set(self.total_equipment))
        self.total_contractor = list(set(self.total_contractor))
        self.total_crewtype = list(set(self.total_crewtype))
        self.total_worktype = list(set(self.total_worktype))
        self.crew_trim_cycle_maintenance = list(set(Helper_Functions.getCrewTrimMainCycles()))
        # Filter out non maintenance cycle trims
        for circuit_id in self.crew_data_dict.keys():
            if circuit_id  in self.useful_circuits:
                self.crew_data_master_relevant[circuit_id] = self.crew_data_dict[circuit_id]
        self.crew_data_dict = {}
        self.crew_data_dict = self.crew_data_master_relevant

    def crewWeekAvgCalc(self,df,len_dates,crew_type):
        df = df[['crewtype','date','unit']]
        df_m = df.groupby(['crewtype','date']).mean().to_dict(orient = 'dict')
        df_c = Counter(df.date)
        total_crews = []
        for crew_date,count in df_c.items():
            try:
                total_crews.append(round(count/df_m['unit'][crew_type,crew_date]))
            except:
                pass
        return(sum(total_crews)/len_dates)
        
        
    def getAvgCrewWeek(self):
        
        for circuit_id, data in self.crew_data_dict.items():
            df = pd.DataFrame(data)
            len_dates = len(set(df.date))
            df_othr = df[(df.crewtype != 'bucket')]
            df_othr = df_othr[(df_othr.crewtype != 'climbing')]
            df_othr['crewtype'] = 'other'
            df_bkt = df[(df.crewtype == 'bucket')]
            df_clmb = df[(df.crewtype == 'climbing')]
            bkt_avg = self.crewWeekAvgCalc(df_bkt,len_dates,'bucket')
            clm_avg = self.crewWeekAvgCalc(df_clmb,len_dates,'climbing')
            oth_avg = self.crewWeekAvgCalc(df_othr,len_dates,'other')
            self.total_crew_per[circuit_id] = {"climbing_per":clm_avg
                                               ,"bucket_per":bkt_avg,
                                                "others_per":oth_avg}
        
        with open(Import_Files.total_crew_per,'wb') as f:
            pickle.dump(self.total_crew_per, f, pickle.HIGHEST_PROTOCOL)
        
        
    def aggregateData(self,previous_trim_data):
        # Create dictionary of lists that aggregates data across all
        # crewtypes and equipment types and calculates contribution as a % of cost
        agg_cost_data_dict = {}
        for i in self.total_crewtype:
            agg_cost_data_dict[i] = 0
        #for i in self.total_equipment:
         #   agg_cost_data_dict[i] = 0
        # For the circut as we parse through the crew samples
        total_cost = 0
        unit_list = []
        for crew_sample in previous_trim_data:
            try:
                agg_cost_data_dict[crew_sample['crewtype']] += crew_sample['cost']
                #agg_cost_data_dict[crew_sample['equipment']] += crew_sample['cost']
                total_cost += float(crew_sample['cost'])
                unit_list.append(crew_sample['unit'])
            except:
                pass

        # Convert equp and crewtype costs to %
        for key in agg_cost_data_dict.keys():
            agg_cost_data_dict[key] = agg_cost_data_dict[key]/total_cost
        agg_cost_data_dict['mean_unit'] = np.mean(unit_list)
        return(agg_cost_data_dict)

    def getFeatures(self):
        output_df = pd.DataFrame()
        for circuit_id in self.crew_data_dict.keys():
            curr_data = self.crew_data_dict[circuit_id]
            curr_agg_data = self.aggregateData(curr_data)
            prev_circuit_id, prev_trim_yr, curr_trim_yr = Helper_Functions.getPrevTrim(circuit_id, self.crew_data_dict)
            if np.isnan(prev_trim_yr):
                prev_agg_data = {'prev_' + key: np.NaN for key in curr_agg_data.keys()}
            else:
                prev_data = self.crew_data_dict[prev_circuit_id]
                prev_agg_data = self.aggregateData(prev_data)
                prev_agg_data = {'prev_' + key: value for key, value in prev_agg_data.items()}
            curr_agg_data = {'curr_' + key: value for key, value in curr_agg_data.items()}
            output_dict = dict(curr_agg_data)
            output_dict.update(prev_agg_data)
            output_dict['circuit_id'] = circuit_id
            output_df = output_df.append(output_dict, ignore_index = True)
        return(output_df)

    def writeFeatures(self,output_df):
        output_df.to_pickle(self.crew_factory_file_path)
