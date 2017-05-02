#!/usr/bin/python
import pandas as pd
import numpy as np
import Step_3_PipelineModel
# Re Run Factories to write files to disk
#Step_3_PipelineModel.createFactoryFiles()
# Load Data From disk
cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data = Step_3_PipelineModel.loadMasterData()
# Join Data Sources
master_df = Step_3_PipelineModel.joinData(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data)
master_df = master_df[master_df.yrs_since_trim > 2]
master_df = master_df[(master_df.curr_total_cost > 5000) & (master_df.oh_mile > 1)]
master_df = master_df.dropna(axis = 0)
# Lists  to store models
scaled_data_models_list = []
unscaled_data_models_list = []
## Build  and predict cost model
#orig_df = master_df
#orig_df = orig_df[~orig_df.index.isin(master_df.index)]
#orig_df.to_csv('dropped.csv')
#X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled,train_circuitid,test_circuitid = Step_3_PipelineModel.preProcessData(master_df,'curr_total_cost',0,drop_feat_list)
master_df['year'] = [x.split('-')[2] for x in master_df.circuit_id]
train_dat = master_df[master_df.year != '2016']
test_dat = master_df[master_df.year == '2016']
drop_feat_list = [col for col in master_df.columns if 'curr' in col]
drop_feat_list.remove('curr_total_cost')
drop_feat_list += ['sub_fed', 'prev_mean_unit', 'src_op_center']
X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled,train_circuits,test_circuits = Step_3_PipelineModel.preProcessData(master_df,'curr_total_cost',0 ,drop_feat_list)
X_train_unscaled = X_train_unscaled.replace([np.inf, -np.inf], 0)
rf_grid_search_cost = Step_3_PipelineModel.buildRandomForestRegCV(X_train_unscaled,y_train_unscaled)
feature_importance = pd.DataFrame()
feature_importance['names'] = X_train_unscaled.columns
feature_importance['importance'] = rf_grid_search_cost.best_estimator_.feature_importances_
print(feature_importance.sort('importance'))
print(rf_grid_search_cost.best_score_)
