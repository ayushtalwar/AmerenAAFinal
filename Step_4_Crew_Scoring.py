#!/usr/bin/python
import pandas as pd
import numpy as np
from scipy import stats
import Import_Files
import Step_2_TimesheetFactory
import Step_2_WeatherFactory
import Step_4_SimilarityEngine
import Step_3_PipelineModel
import datetime
import csv
def calculateScore(row, circuit_distributions):
	circuit = row['circuit_id']
	crewtype = row['CREWTYPE']
	trims = row['NOOFTRIMS']
	removals = row['NOOFREMOVALS']
	brush = row['BRUSHACRES']
	totalhours = row['TOTALHOURS']
	if circuit in circuit_distributions:
		if crewtype in circuit_distributions[circuit]:
			dist, standard = circuit_distributions[circuit][crewtype]
			hrs_per_trim, hrs_per_rem, hrs_per_brush = standard
			standard_hours = trims * hrs_per_trim + removals * hrs_per_rem + brush * hrs_per_brush
			if totalhours == 0:
				return np.NaN
			elif len(dist) == 0:
				return np.NaN
			else:
				efficiency = float(standard_hours) / totalhours
				score = stats.percentileofscore(dist, efficiency)
				return score
		else:
			return np.NaN
	else:
		return np.NaN
year = str(datetime.datetime.now().year)
print("Loading timsheet data")
tsf = Step_2_TimesheetFactory.TimesheetFactory(Import_Files.timesheet_f, Import_Files.ts_factory_f, Import_Files.carryover_f)
days_for_scoring = tsf.getDaysForScoring(year)
circuit_ids = list(set(days_for_scoring['circuit_id']))
# Get inclement weather days
print("Getting inclement weather days")
wf = Step_2_WeatherFactory.WeatherFactory(Import_Files.cth_factory_file_f,Import_Files.total_weather_data,
                                          Import_Files.weather_factory_f,Import_Files.weather_coordinates_f,
                                          Import_Files.ss_coordinates_f)
inclement_df = wf.getInclementDays(circuit_ids, year)
days_for_scoring = days_for_scoring.merge(inclement_df, how='left', on=['circuit_id', 'LABDATE'])
days_for_scoring['MO_WEATHER_TYPE_SCORE'] = days_for_scoring['isInclement'].fillna(0)
print("Scoring production")
# Perform scoring
prev_trims = pd.read_csv(Import_Files.prev_trim_f)
prev_trims = {circuit_id: prev_id for circuit_id, prev_id in zip(prev_trims['circuit_id'], prev_trims['prev_circuit_id'])}
feature_list = ['circuit_id','curr_cust_per_mi','curr_rem_per_mi', 'curr_trims_per_mi',
			   'curr_brush_per_mi', 'curr_rem_hr_per_mi', 'curr_trim_hr_per_mi','curr_brush_hr_per_mi',
			   'curr_tot_hr_per_mi', 'curr_bucket', 'curr_climbing', 'curr_helicopter',
			   'curr_jarraff', 'otgs_snce_lst_trim', 'Canopy cover', 'oh_mile',
			   'yrs_since_trim', 'curr_trims_per_hr']
engine = Step_4_SimilarityEngine.SimilarityEngine(feature_list, run_factories=False)
circuit_distributions = {}
for circuit in sorted(circuit_ids):
    if circuit in prev_trims:
        prev_id = prev_trims[circuit]
        print(circuit)
        try:
            similar = engine.getOpSimilar(prev_id)
            circuit_distributions[circuit] = tsf.getScoreDistributions(similar, 5)
        except:
            try:
                similar = engine.getSimilar(prev_id)
                circuit_distributions[circuit] = tsf.getScoreDistributions(similar, 5)
            except:
                pass
    else:
        print("Missing: " + circuit)        
scored_days = days_for_scoring.copy()
scored_days['MO_PRODUCTIVITY_SCORE'] = scored_days.apply(lambda row: calculateScore(row, circuit_distributions), axis=1)
scored_days = scored_days[['LINENAME', 'PECONTRACTOR', 'CREWTYPE', 'CREW_NO_1', 'LABDATE', 'MO_PRODUCTIVITY_SCORE', 'MO_WEATHER_TYPE_SCORE']]
sql_flg = 0
with open(Import_Files.sql_flag) as f:
    csv_f = csv.reader(f)
    for row in csv_f:
        if row[0] == '1':
            sql_flg = 1
if sql_flg == 1:
    scored_days.to_sql('MO_CREW_WORK_DETAILS', Import_Files.engine, if_exists='replace')
else:    
    scored_days.to_csv(Import_Files.path_to_data +  "MO_CREW_WORK_DETAILS.csv",index = False)
  


