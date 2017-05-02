# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:21:08 2017

@author: Q141541
"""
# Applying the SOTT filter
# Removing extraneous columns
# Filling null values
# Aggregating up to the line/TTO/timesheet week level
# What was the intent of the drop of the unit column?
# Outputting a single CSV for each input

import os
import pandas as pd
import glob
import csv
import re

# Convert to argsv
files = list(glob.glob("*COST*"))
total_data = pd.DataFrame()
output_filename = "timesheet_data.csv"
crew_trim_history_master = {}
for i in files:
    tto_linenames = {}
    print(i)
    data = pd.read_csv(i)
    data['ACTIVITY'] = data['ACTIVITY'].map(lambda x: x if type(x)!=str else x.lower())
    data = data[data['ACTIVITY'] == 'sott']
    data = data[["LINENAME","TTO","LABDATE","NONPRODHOURS",
                  "NOQTRSPANS", "BRUSHACRES", "NOOFREMOVALS", "NOOFTRIMS",
                  "QTRSPANHOURS", "BRUSHHOURS",
                  "REMOVALHOURS", "TRIMHOURS","TOTALCOST","CREW_NO_1","CREWTYPE","PECONTRACTOR"]]
    data = data.fillna(0)
    total_data = total_data.append(data)

total_data.to_csv(output_filename, index=False)
