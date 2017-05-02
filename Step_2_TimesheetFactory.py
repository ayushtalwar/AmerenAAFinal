#!/usr/bin/python
import numpy as np
import pandas as pd
import Import_Files
import Helper_Functions
import csv
pd.options.mode.chained_assignment = None  # default='warn'

class TimesheetFactory():
    def __init__(self, timesheet_f, ts_factory_f, carryover_f):
        self.ts_factory_f = ts_factory_f
        self.timesheet_f = timesheet_f
        self.carryovers = dict.fromkeys(pd.read_csv(carryover_f)['circuit_id'])
        self.circuit_list = []
        self.ts_df = pd.DataFrame()
        self.climbing_sum_df = pd.DataFrame()
        self.bucket_sum_df = pd.DataFrame()
        self.loadRaw()
        self.calcSums()

    def calcSums(self):
        self.climbing_sum_df = self.ts_df[self.ts_df['CREWTYPE'] == 'climbing'].groupby(['circuit_id'], as_index=False).agg({'NOOFTRIMS': 'sum',
                                                                                      'NOOFREMOVALS': 'sum',
                                                                                      'BRUSHACRES': 'sum',
                                                                                      'TRIMHOURS': 'sum',
                                                                                      'REMOVALHOURS': 'sum',
                                                                                      'BRUSHHOURS': 'sum'})
        self.bucket_sum_df = self.ts_df[self.ts_df['CREWTYPE'] == 'bucket'].groupby(['circuit_id'], as_index=False).agg({'NOOFTRIMS': 'sum',
                                                                                      'NOOFREMOVALS': 'sum',
                                                                                      'BRUSHACRES': 'sum',
                                                                                      'TRIMHOURS': 'sum',
                                                                                      'REMOVALHOURS': 'sum',
                                                                                      'BRUSHHOURS': 'sum',
                                                                                      'TOTALHOURS': 'sum'})
        return

    def extractYearFromDate(self, date):
        return str(date)[0:4]

    def extractYearFromID(self, circuit_id):
        if pd.isnull(circuit_id):
            return np.NaN
        else:
            return circuit_id[-4:]

    def correctBrush(self, brushacres):
        # Convert sq. feet to acres
        if brushacres > 5:
            brushacres = float(brushacres) / 43560
        return brushacres

    def correctCarryovers(self, circuit_id):
        if circuit_id in self.carryovers:
            sub, fed, year = circuit_id.split('-')
            year = str(int(year) + 1)
            circuit_id = sub + '-' + fed + '-' + year
        return circuit_id

    def loadRaw(self):
        sql_flg = 0
        with open(Import_Files.sql_flag) as f:
            csv_f = csv.reader(f)
            for row in csv_f:
                if row[0] == '1':
                    sql_flg = 1
                 
        if sql_flg == 1:
            df = pd.read_sql(Import_Files.timesheet_table,Import_Files.engine)
        else:    
            df = pd.read_csv(self.timesheet_f)        
        df['YEAR'] = df['LABDATE'].apply(self.extractYearFromDate)
        df['LINENAME'] = df['LINENAME'].apply(Helper_Functions.substationFeederStandardizer)
        df['FLAG'] = 0
        df['FLAG'][df['BRUSHACRES'] > 5] = 1
        df['BRUSHACRES'] = df['BRUSHACRES'].apply(self.correctBrush)
        df['CREWTYPE'] = df['CREWTYPE'].apply(Helper_Functions.standardizeCrewType)
        df['TOTALHOURS'] = df['TRIMHOURS'] + df['REMOVALHOURS'] + df['BRUSHHOURS']
        df['circuit_id'] = df['LINENAME'] + '-' + df['YEAR']
        df['circuit_id'] = df['circuit_id'].apply(self.correctCarryovers)
        df['YEAR'] = df['circuit_id'].apply(self.extractYearFromID)
        df.index = df['circuit_id']
        df = df.dropna()
        self.ts_df = df.fillna(0)
        return

    def getScoreDistributions(self, circuit_list, num_similar):
        distributions = {}
        distributions['climbing'] = self.getScores(circuit_list, 'climbing', num_similar)
        distributions['bucket'] = self.getScores(circuit_list, 'bucket', num_similar)
        return distributions

    def setComparables(self, circuit_list, crew_type, num_similar):
        count = 0
        while count < 200:
            reduced_list = circuit_list[0:num_similar]
            count = len(self.ts_df[(self.ts_df['circuit_id'].isin(reduced_list)) & (self.ts_df['CREWTYPE'] == crew_type)].index)
            num_similar += 10
        return reduced_list

    def getScores(self, circuit_list, crew_type, num_similar):
        circuit_list = self.setComparables(circuit_list, crew_type, num_similar)
        if crew_type == 'climbing':
            sums = self.climbing_sum_df
        else:
            sums = self.bucket_sum_df
        sums = sums[sums['circuit_id'].isin(circuit_list)].sum(numeric_only=True)
        hrs_per_trim = self.getManHrsPerUnit(sums, 'TRIMHOURS', 'NOOFTRIMS')
        hrs_per_rem = self.getManHrsPerUnit(sums, 'REMOVALHOURS', 'NOOFREMOVALS')
        hrs_per_brush = self.getManHrsPerUnit(sums, 'BRUSHHOURS', 'BRUSHACRES')
        scores_df = self.ts_df[(self.ts_df['circuit_id'].isin(circuit_list)) & (self.ts_df['CREWTYPE'] == crew_type)].copy()
        scores_df['score'] = scores_df['NOOFTRIMS'] * hrs_per_trim + scores_df['NOOFREMOVALS'] * hrs_per_rem + scores_df['BRUSHACRES'] * hrs_per_brush
        scores_df['score'] = (scores_df['score'] / scores_df['TOTALHOURS']).fillna(0)
        standard = (hrs_per_trim, hrs_per_rem, hrs_per_brush)
        return list(scores_df['score']), standard

    def getManHrsPerUnit(self, df, hours_str, units_str):
        hours = df[hours_str]
        units = df[units_str]
        if units == 0:
            return 0
        else:
            return hours / units

    def getDaysForScoring(self, year):
        return self.ts_df[(self.ts_df['YEAR'] == year)  & (self.ts_df['FLAG'] == 0) ].copy()

