#!/usr/bin/python
from sqlalchemy import create_engine
   
path_to_data = "/Users/Ayush Talwar/Documents/Ameren/Data/"

engine = create_engine('sqlite:///example.db')
circuit_characteristic_file = path_to_data +  'Ameren_circuit_data.xlsx'
weather_coordinates_f =  path_to_data +  'Weather_Coordinates.csv'
ss_coordinates_f =  path_to_data + 'SS_Coordinates.csv'
total_cleaned_cth_data_input = path_to_data +  'total_cleaned_data.csv'
outage_data_f =  path_to_data +'vegetation_outages.csv'
crew_data_input = path_to_data + 'crew_data_cat.csv'
il_trim_list =  path_to_data +'Circuit_Schedule_type_IL.csv'
mo_trim_list = path_to_data +'Circuit_Schedule_type_MO.csv'
total_weather_data = path_to_data + "total_weather_data_2.pickle"
circuit_char_factory_f = path_to_data + 'circuit_char_factory_f.pickle'
cth_factory_file_f =path_to_data +'cth_factory_f.pickle'
weather_factory_f =  path_to_data +'weather_factory_f.pickle'
crew_factory_f = path_to_data +'crew_factory_f.pickle'
outage_factory_f =path_to_data +'outage_factory_f.pickle'
latest_master_df = path_to_data +'latest_master_df.csv'
gis_factory_f =  path_to_data +'gis_factory_f.pickle'
gis_data_f = path_to_data +'Canopy_out.csv'
timesheet_f =path_to_data + 'timesheet_data.csv'
ts_factory_f =path_to_data +  'ts_factory_f.pickle'
carryover_f = path_to_data + 'carryover.csv'
prev_trim_f = path_to_data + 'prev_trim.csv'
last_trim_month_collector = path_to_data + 'last_trim_month_collector.pkl'
path_to_total_rel_df =path_to_data + 'total_rel.pickle'
total_crew_per = path_to_data +'total_crew_per.pkl'
useful_circuits =  path_to_data +'useful_circuits.pkl'
reliability_train_data = path_to_data + "reliability_X_Train.pickle"
test_dat_disk = path_to_data + 'test_dat_disk.pickle'
what_if_output = path_to_data + 'what_if_output.json'
sql_flag =  path_to_data + "sql_flag.csv"
weather_table = 'weather'
timesheet_table = 'timesheet'
outages_table = 'outages'
gis_table = 'gis'
circuit_characteristics_table = 'circuit_char'
crew_trim_history_table = 'crew_trim_history_table'
weather_download_url = path_to_data +'Dwonload_URL_withFormat.txt'

