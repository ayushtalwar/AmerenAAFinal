#!/usr/bin/python
import re
import pandas as pd
import numpy as np
import Import_Files
import Helper_Functions
import csv

class OutagesFactory():
    def __init__(self, outage_data_f,cth_factory_file_f,outage_factory_f):
        # Define the file from path location
        # Create Pandas dataframe from csv
        sql_flg = 0
        with open(Import_Files.sql_flag) as f:
            csv_f = csv.reader(f)
            for row in csv_f:
                if row[0] == '1':
                    sql_flg = 1
                 
        if sql_flg == 1:
            self.outages = pd.read_sql(Import_Files.outages_table,Import_Files.engine)
        else:    
            self.outages = pd.read_csv(outage_data_f ,engine = 'python')    
        # CMI calculation before group-by statement
        self.outages['CMI'] = self.outages['MINS_SERV_OUT']
        # Create DataFrame at Circuit Year level of summarized stats
        self.outages = self.outages.groupby(['SUPPLY_FDR_CURR','YEAR'], as_index=False).agg({'CMI': ['sum'], 'OAS_ORDER_NO': ['count']}).rename(columns = {'OAS_ORDER_NO':'NUMBER_OF_OUTAGES'})
        # Clean circuit ids
        self.outages['SUPPLY_FDR_CURR'] = self.outages['SUPPLY_FDR_CURR'].apply(self.clean_circuit_id)
        self.cth_data = pd.read_pickle(cth_factory_file_f)
        self.circuit_trim_list = pd.DataFrame(self.cth_data['circuit_id'])
        self.outage_factory_f = outage_factory_f
        self.all_trims_list = Helper_Functions.getCrewTrimMainCycles()

    def clean_circuit_id(self, raw_id):
        if raw_id == np.nan or len(raw_id) > 6 or re.search('[^A-Za-z0-9]', raw_id):
            return '000-000'
        elif len(raw_id) < 6:
            raw_id = raw_id.zfill(6)
        return raw_id[:3] + "-" + raw_id[3:]

    def count_outages(self, begin_year, end_year, circuit):
        filter_rows = self.outages[(self.outages.YEAR >= begin_year) & (self.outages.YEAR < end_year) & (self.outages.SUPPLY_FDR_CURR == circuit)]
        cmi_sum = filter_rows['CMI'].values.sum()
        num_outages_sum = filter_rows['NUMBER_OF_OUTAGES'].values.sum()
        return (cmi_sum, num_outages_sum)

    def getFeatures(self):
        noutages_list = []
        cmi_list = []
        for row in self.circuit_trim_list.iterrows():
            (sub, feeder, year) = row[1]['circuit_id'].split("-")
            sub_feeder = sub + "-" + feeder
            # build list of previous trims
            trims_on_same_sub_fed = [x for x in self.all_trims_list if x.startswith(sub_feeder)]
            trims_on_same_sub_fed = [int(x.split('-')[2]) for x in trims_on_same_sub_fed if int(x.split('-')[2]) < int(year)]
            (cmi, noutages) = np.NaN, np.NaN
            if len(trims_on_same_sub_fed) > 0:
                last_trim_yr = max(trims_on_same_sub_fed)
                if last_trim_yr > 2007:
                    (cmi, noutages) = self.count_outages(last_trim_yr, int(year), sub_feeder)
            noutages_list.append(noutages)
            cmi_list.append(cmi)

        self.circuit_trim_list = self.circuit_trim_list.assign(otgs_snce_lst_trim=noutages_list)
        self.circuit_trim_list = self.circuit_trim_list.assign(cmi_snce_lst_trim=cmi_list)
        return(self.circuit_trim_list)

    def writeFeatures(self, circuit_trim_list):
        circuit_trim_list.to_pickle(self.outage_factory_f)
