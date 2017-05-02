#!/usr/bin/python
import pandas as pd
import operator
import numpy as np
import Import_Files
import csv

class WeatherFactory():
    def __init__(self, cth_factory_file_f, total_weather_data, weather_factory_f,
                     weather_coordinates_f, ss_coordinates_f):
        sql_flg = 0
        with open(Import_Files.sql_flag) as f:
            csv_f = csv.reader(f)
            for row in csv_f:
                if row[0] == '1':
                    sql_flg = 1
                 
        if sql_flg == 1:
            weather_data = pd.read_sql(Import_Files.weather_table,Import_Files.engine)
        else:    
            weather_data = pd.read_pickle(Import_Files.total_weather_data)
       
        self.cth_data = pd.read_pickle(cth_factory_file_f)
        self.weather_coordinates = pd.read_csv(weather_coordinates_f)
        self.substation_coordinates = pd.read_csv(ss_coordinates_f)
        self.weather_data = self.scrubWeatherData(weather_data)
        self.weather_factory_f = weather_factory_f
        weather_stn = list(set(pd.Series(weather_data['Weather_Stn']).dropna()))
        self.weather_coordinates =  self.weather_coordinates[['WTHRSTATIONID','LATITUDE','LONGITUDE']]
        self.weather_coordinates = self.weather_coordinates[self.weather_coordinates['WTHRSTATIONID'].isin(weather_stn)]
        self.sub_weather_lookup = self.weatherSubstationLookup()
    def getFeatures(self):
        intervals = self.getNeededIntervals(self.cth_data, self.sub_weather_lookup)
        interval_features = self.calcFeatures(intervals, self.weather_data)
        self.weather_df = self.buildWeatherDF(interval_features, self.cth_data, self.sub_weather_lookup)
        return self.weather_df

    def writeFeatures(self):
        weather_df_col = self.weather_df.columns
        weather_df_col = [x for x in weather_df_col if not x.startswith('LAB')]
        self.weather_df = self.weather_df[weather_df_col]
        self.weather_df.to_pickle(self.weather_factory_f)

    def weatherSubstationLookup(self):
        weather_sub_master = {}
        sub_weather_lookup = {}
        self.substation_coordinates = self.substation_coordinates.drop('Sub Name',1)
        self.substation_coordinates = self.substation_coordinates.dropna(axis = 0)
        self.substation_coordinates.columns.values[0] = 'substation'
        for s in self.substation_coordinates.iterrows():
            sub_lat = (s[1]['Latitude'])
            sub_lon = (s[1]['Longitude'])
            for w in self.weather_coordinates.iterrows():
                weather_sub_lat = w[1]["LATITUDE"]
                weather_sub_lon = w[1]["LONGITUDE"]
                d = self.coordinateDistCalc(sub_lat,sub_lon,weather_sub_lat,weather_sub_lon)
                # For every substation calculate distance to all weather stations
                sub_weather_lookup[w[1]['WTHRSTATIONID']] = d
                # Pick the closest weather station
            closest_ws = min(sub_weather_lookup.items(), key=operator.itemgetter(1))[0]
            sub  = s[1]['substation']
            if(len(sub) == 1):
                sub = '00' + sub
            if (len(sub)==2):
                sub = '0' + sub

            if closest_ws in weather_sub_master.keys():
                weather_sub_master[closest_ws].append(sub)
            else:
                weather_sub_master[closest_ws] = [sub]
        # invert lookup
        sub_weather_lookup = {}
        for ws, subs in weather_sub_master.items():
            for sub in subs:
                sub_weather_lookup[sub] = ws
        return sub_weather_lookup

    def coordinateDistCalc(self, lat1,lon1,lat2,lon2):
        from geopy.distance import vincenty
        a = (lat1,lon1)
        b = (lat2,lon2)
        return(vincenty(a,b).miles)

    def encode(self, input_string):
        count = 1
        prev = ''
        lst = []
        for character in input_string:
            if character != prev:
                if prev:
                    entry = (prev,count)
                    lst.append(entry)
                    #print lst
                count = 1
                prev = character
            else:
                count += 1
        else:
            entry = (character,count)
            lst.append(entry)
        return lst

    def getContinuousDays(self, relevant_weather_data,year,span,feature, run_length_days,threshold):
        relevant_weather_data = relevant_weather_data[relevant_weather_data['year'] <= year]
        relevant_weather_data = relevant_weather_data[relevant_weather_data['year'] >= year - span]
        days_above_threshold = 0
        if relevant_weather_data.shape[0] > 0:
            relevant_weather_data = relevant_weather_data[['Date',feature]]
            relevant_weather_data = relevant_weather_data.dropna().sort_values('Date')
            relevant_weather_data['Target'] = 1
            relevant_weather_data.loc[relevant_weather_data[feature]<=threshold, 'Target'] = 2
            rle_feature = self.encode(relevant_weather_data['Target'])
            for item in rle_feature:
                if item[0] == 2 and item[1] > run_length_days:
                    days_above_threshold +=1
        return(days_above_threshold)

    def getNeededIntervals(self, cth_data, sub_weather_lookup):
        # Maps weather station intervals to relevant circuit ids
        intervals = {}
        for row in cth_data.iterrows():
            circuit_id = row[1]['circuit_id']
            sub,feeder,year = str.split(circuit_id,'-')
            end_year = int(year)
            # Should compute the exact cycle at some point
            start_year = (end_year - 4)
            if sub in sub_weather_lookup:
                ws = sub_weather_lookup[sub]
                if ws not in intervals:
                    intervals[ws] = set()
                intervals[ws].add((start_year, end_year))
        return intervals

    def getColNames(self, weather_data, stats_cols, sme_cols):
        weather_cols = weather_data.drop(['Date','year','LABDATE','Weather_Stn','Events'],1).columns
        col_names = []
        for wcol in weather_cols:
            for scol in stats_cols:
                col_names.append(wcol + ' ' + scol)
        col_names = col_names + sme_cols
        return col_names

    def calcFeatures(self, intervals, weather_data):
        interval_features = {}
        stats_cols = ['mean','std','min','max','25%','50%','75%']
        sme_cols = ['Drghts_lst_2_yrs', 'Drghts_lst_3_4_yrs', 'Frzing_wks']
        feature_cols = self.getColNames(weather_data, stats_cols, sme_cols)
        for ws in intervals.keys():
            ws_data = weather_data[weather_data['Weather_Stn'] == ws]
            for (start_year, end_year) in intervals[ws]:
                interval_data = ws_data[(ws_data['year'] <= end_year) &
                                        (ws_data['year'] >= start_year)]
                drought_Last_2_Years = self.getContinuousDays(interval_data, end_year, 2, "PrecipitationIn", 15, 0.5)
                drought_Last_3_4_Years = self.getContinuousDays(interval_data, end_year - 2, 2, "PrecipitationIn", 15, 0.5)
                freeze_wks = self.getContinuousDays(interval_data, end_year, 4, "TemperatureF", 7, 33)
                summary_stats = interval_data.drop(['Date','year','Weather_Stn','Events'],1)
                summary_stats = summary_stats.dropna(0)
                summary_stats = summary_stats.describe().T
                features = np.matrix(np.append(summary_stats[stats_cols].values.flatten(),
                                     [drought_Last_2_Years, drought_Last_3_4_Years, freeze_wks]))
                interval_features[(ws, start_year, end_year)] = pd.DataFrame(features, columns=feature_cols)
        return interval_features

    def convertToLabdate(self, date):
        if pd.isnull(date):
            return np.NaN
        return int(str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2))

    def scrubWeatherData(self, weather_data):
        weather_data['Date'] = pd.to_datetime(weather_data['CST'])
        weather_data['LABDATE'] = weather_data['Date'].apply(self.convertToLabdate)
        weather_data = weather_data.drop('CST',1)
        weather_data['year'] = weather_data['Date'].dt.year
        weather_data = weather_data[[' CloudCover',' Events',' Max Gust SpeedMPH',
                                     ' Max Sea Level PressureIn', ' Max VisibilityMiles',' Max Wind SpeedMPH',
                                     'Max Dew PointF', 'Max Humidity',
                                     'Max TemperatureF','PrecipitationIn','Weather_Stn','Date','year','LABDATE']]
        weather_data = weather_data.rename(index=str, columns={' CloudCover': 'CloudCover', ' Events': 'Events',
                                                               ' Max Gust SpeedMPH': 'Gust SpeedMPH',
                                                               ' Max Sea Level PressureIn': 'Sea Level PressureIn',
                                                               ' Max VisibilityMiles': 'VisibilityMiles',
                                                               ' Max Wind SpeedMPH': 'Wind SpeedMPH',
                                                               'Max Dew PointF': 'Dew PointF', 'Max Humidity': 'Humidity',
                                                               'Max TemperatureF': 'TemperatureF'})
        weather_data['PrecipitationIn'] = pd.to_numeric(weather_data['PrecipitationIn'], errors='coerce').fillna(0.0)
        return weather_data

    def buildWeatherDF(self, interval_features, cth_data, sub_weather_lookup):
        weather_df = pd.DataFrame()
        for row in cth_data.iterrows():
            circuit_id = row[1]['circuit_id']
            sub, feeder, year = str.split(circuit_id,'-')
            end_year = int(year)
            # Should compute the exact cycle at some point
            start_year = (end_year - 4)
            if sub in sub_weather_lookup:
                ws = sub_weather_lookup[sub]
                row = interval_features[(ws, start_year, end_year)]
                row['circuit_id'] = circuit_id
                weather_df = weather_df.append(row)
        weather_df = weather_df.drop_duplicates()
        return weather_df

    def testInclement(self, row):
        isInclement = 0
        if row['TemperatureF'] >= 90:
            isInclement = 1
        elif row['TemperatureF'] <= 20:
            isInclement = 1
        elif row['PrecipitationIn'] >= 0.33:
            isInclement = 1
        return isInclement

    def getWeatherStation(self, circuit_id):
        sub, fed, year = circuit_id.split('-')
        if sub in self.sub_weather_lookup:
            return self.sub_weather_lookup[sub]
        else:
            return np.NaN

    def getInclementDays(self, circuit_ids, year):
        mapping_df = pd.DataFrame()
        mapping_df['circuit_id'] = circuit_ids
        mapping_df['Weather_Stn'] = mapping_df['circuit_id'].apply(self.getWeatherStation)
        inclement_df = self.weather_data[self.weather_data['year'] == float(year)].copy()
        inclement_df['isInclement'] = inclement_df.apply(self.testInclement, axis=1)
        inclement_df = inclement_df[inclement_df['isInclement'] == 1]
        inclement_df = inclement_df[['Weather_Stn', 'LABDATE', 'isInclement']]
        mapping_df = mapping_df.merge(inclement_df, how='inner', left_on='Weather_Stn', right_on='Weather_Stn')
        mapping_df = mapping_df.drop('Weather_Stn', axis=1)
        return mapping_df

