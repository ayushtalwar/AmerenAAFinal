#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:30:26 2017

@author: Ayush Talwar
"""

import pandas as pd
import io
import requests
import Import_Files
import numpy as np
fname = (Import_Files.weather_download_url)
with open(fname) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 
current_weather_stn = 0
total_weather_data = pd.DataFrame()
for i in range(0,len(content)):
    item = content[i]
    
    if len(item) == 3:
        current_weather_stn = item
        i += 1
    
    elif len(item) > 3:
        url = item
        s=requests.get(url).content
        df=pd.read_csv(io.StringIO(s.decode('utf-8')))
        df['Weather_Stn'] = current_weather_stn
        total_weather_data = total_weather_data.append(df)

total_weather_data.CST = pd.to_datetime(total_weather_data.CST)
total_weather_data.CDT = pd.to_datetime(total_weather_data.CDT)
dates = []
for row in total_weather_data.iterrows():
    cst = str(row[1]['CST'])
    cdt = str(row[1]['CDT'])
    if cst == 'nan':
        dates.append(cdt)
    else:
        dates.append(cst)
total_weather_data['CST'] = dates 
total_weather_data  = total_weather_data.drop(['CDT','EST','EDT'],1)                  
total_weather_data.to_pickle( Import_Files.total_weather_data)        
    
    
       
 