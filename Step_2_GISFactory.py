#!/usr/bin/python
import re
import pandas as pd
import numpy as np
import Import_Files
import csv
class GISFactory():
    def __init__(self, gis_data_f, gis_factory_f):
        self.gis_factory_f = gis_factory_f
        sql_flg = 0
        with open(Import_Files.sql_flag) as f:
            csv_f = csv.reader(f)
            for row in csv_f:
                if row[0] == '1':
                    sql_flg = 1
                 
        if sql_flg == 1:
            gis_data = pd.read_sql(Import_Files.gis_table,Import_Files.engine)
        else:    
            gis_data = pd.read_csv(gis_data_f, engine = 'python')        
        
        gis_data['PHASE_CONFIG'] = gis_data['PHASE_CONFIG'].fillna('UNKNOWN')
        gis_df = pd.DataFrame()
        gis_df['sub_fed'] = gis_data['FEEDER2_NUM'].apply(self.clean_sub_fed)
        gis_df['Line phases'] = gis_data['PHASE_CONFIG'].apply(self.extract_phases)
        gis_df['Phase orientation'] = gis_data['PHASE_CONFIG'].apply(self.extract_phase_orientation)
        gis_df['Canopy cover'] = gis_data['RASTERVALU']
        self.grouped_df = gis_df.groupby('sub_fed', as_index=False).agg({'Line phases': lambda x: x.value_counts().index[0],
                                                       'Phase orientation': lambda x: x.value_counts().index[0],
                                                       'Canopy cover': np.mean})
        dummies = pd.get_dummies(self.grouped_df['Phase orientation'])
        self.grouped_df = self.grouped_df.drop('Phase orientation', 1)
        self.grouped_df = self.grouped_df.join(dummies)

    def extract_phases(self, phase_str):
        if phase_str == 'UNKNOWN':
            return 1
        else:
            return int(phase_str[0:1])

    def extract_phase_orientation(self, phase_str):
        try:
            phases, orientation, ordering = phase_str.split(',')
            if 'HORIZONTAL' in orientation:
                coded_orientation = 'Horizontal'
            elif 'VERTICAL' in orientation:
                coded_orientation = 'Vertical'
            else:
                coded_orientation = 'Other'
        except:
            coded_orientation = 'Horizontal'
        return coded_orientation

    def clean_sub_fed(self, raw_id):
        if raw_id == np.nan or len(raw_id) > 6 or re.search('[^A-Za-z0-9]', raw_id):
            return '000000'
        elif len(raw_id) < 6:
            raw_id = raw_id.zfill(6)
        return raw_id

    def getFeatures(self):
         return(self.grouped_df)

    def writeFeatures(self):
        self.grouped_df.to_pickle(self.gis_factory_f)
