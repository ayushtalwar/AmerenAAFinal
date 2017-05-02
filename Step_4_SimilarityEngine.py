#!/usr/bin/python
import pdb
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

import Step_3_PipelineModel

class SimilarityEngine():
	def __init__(self, feature_list, run_factories=False):
		if run_factories:
			# Re Run Factories to write files to disk
			Step_3_PipelineModel.createFactoryFiles()
		# Load Data From disk
		cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data = Step_3_PipelineModel.loadMasterData()
		# Join Data Sources
		master_df = Step_3_PipelineModel.joinData(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data)
		self.src_op_center = master_df['src_op_center']
		self.src_op_center.index = master_df['circuit_id']
		master_df = master_df[feature_list]
		master_df.index = master_df.circuit_id
		master_df = master_df.drop('circuit_id',1)
		# Impute missing values
		master_df = self.impute(master_df)
		scaler = StandardScaler().fit(master_df)
		scaled_df = scaler.transform(master_df)
		distances = sklearn.metrics.pairwise.pairwise_distances(scaled_df, metric='euclidean', n_jobs=1)
		self.distances = pd.DataFrame(distances, index=master_df.index, columns=master_df.index)

	def impute(self, df):
		df = df.replace([-np.inf, np.inf], np.nan)
		for column in df.columns:
			median = df[column].median()
			df[column] = df[column].fillna(median)
		return df

	def getSimilar(self, circuit_id):
		return list(self.distances[circuit_id].sort_values().index)

	def getOpSimilar(self, circuit_id):
		similar = self.distances[circuit_id]
		similar = pd.concat([similar, self.src_op_center], axis=1)
		try:
			op_center = self.src_op_center.loc[circuit_id]
			if not pd.isnull(op_center):
				op_similar = similar[similar['src_op_center'] == op_center]
				op_similar = op_similar[circuit_id].sort_values().index
				nonop_similar = similar[similar['src_op_center'] != op_center]
				nonop_similar = nonop_similar[circuit_id].sort_values().index
				return list(op_similar.append(nonop_similar))
		except:
			pass
		return list(similar[circuit_id].sort_values().index)
