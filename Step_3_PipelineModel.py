#!/usr/bin/python
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.feature_extraction import DictVectorizer
import Import_Files
import  Step_2_CircuitCharacteristicsFactory
import  Step_2_CrewDataFactory
import  Step_2_CrewTrimHistoryFactory
import  Step_2_Outages_factory
import  Step_2_WeatherFactory
import  Step_2_GISFactory
def splitCircuitColumn(df,feature,ch):
    subs,feeders = [],[]
    for i in df.iterrows():
        sub,feeder,year = str.split(i[1][feature],ch)
        subs.append(sub)
        feeders.append(feeder)
    df['substation_number'] = subs
    df['feeder'] = feeders
    df['sub_fed'] = df['substation_number'] + df['feeder']
    df = df.drop(['substation_number','feeder'],1)
    return(df)
def createFactoryFiles():
    print("Trim History")
    Step_2_CrewTrimHistoryFactory.CrewTrimHistoryFactory(Import_Files.total_cleaned_cth_data_input,
                                                         Import_Files.cth_factory_file_f,
                                                         Import_Files.carryover_f,Import_Files.last_trim_month_collector).writeFeatures()
   
    print("Circuit Characteristics")
    Step_2_CircuitCharacteristicsFactory.CircuitCharacteristicsFactory(Import_Files.circuit_characteristic_file,Import_Files.circuit_char_factory_f).writeFeatures()
    print("Weather")
    weath_obj = Step_2_WeatherFactory.WeatherFactory(Import_Files.cth_factory_file_f,Import_Files.total_weather_data,
                                                     Import_Files.weather_factory_f,Import_Files.weather_coordinates_f,
                                                     Import_Files.ss_coordinates_f)
    weath_obj.getFeatures()
    weath_obj.writeFeatures()
    print("Crew Data")
    useful_circuits = list(pd.read_pickle(Import_Files.cth_factory_file_f).circuit_id)
    crew_data_obj = Step_2_CrewDataFactory.CrewDataFactory(Import_Files.crew_data_input,
                                                           Import_Files.crew_factory_f,useful_circuits)
    crew_data_feat = crew_data_obj.getFeatures()
    crew_data_obj.writeFeatures(crew_data_feat)
    print("Outages")
    outage_object_obj = Step_2_Outages_factory.OutagesFactory(Import_Files.outage_data_f,
                                                              Import_Files.cth_factory_file_f,
                                                              Import_Files.outage_factory_f)
    outage_object_feat = outage_object_obj.getFeatures()
    outage_object_obj.writeFeatures(outage_object_feat)
    print("GIS")
    gis_factory = Step_2_GISFactory.GISFactory(Import_Files.gis_data_f,
                                               Import_Files.gis_factory_f)
    gis_factory.writeFeatures()

def loadMasterData():
    #Crew trim history data
    cth_data = pd.read_pickle(Import_Files.cth_factory_file_f)
    cth_data  = splitCircuitColumn(cth_data,'circuit_id','-')
    #Circuit characteristics data
    circuit_char_data = pd.read_pickle(Import_Files.circuit_char_factory_f)
    #Weather data
    weather_data = pd.read_pickle(Import_Files.weather_factory_f)
    weather_data = splitCircuitColumn(weather_data,'circuit_id','-')
    #Outage data
    outage_data = pd.read_pickle(Import_Files.outage_factory_f)
    # Crew data
    crew_data = pd.read_pickle(Import_Files.crew_factory_f)
    # GIS data
    gis_data = pd.read_pickle(Import_Files.gis_factory_f)
    return(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data)

def joinData(cth_data,circuit_char_data,weather_data,outage_data,crew_data,gis_data):
    ## Join data sources that use substation and feeder as a key
    master_df = pd.DataFrame()
    master_df = cth_data.merge(circuit_char_data, on = ['sub_fed'], how = 'left')
    master_df = master_df.merge(gis_data,on = ['sub_fed'],how = 'left')
    master_df = master_df.drop(['sub_fed'],1)
    ## Join data sources that use substation and feeder and year or circuit id as a key
    master_df = master_df.merge(weather_data, on = ['circuit_id'],how = 'inner')
    master_df = master_df.merge(outage_data,on = ['circuit_id'],how = 'left')
    master_df = master_df.merge(crew_data,on = ['circuit_id'],how = 'left')
    master_df = calcDerivedFeatures(master_df, 'prev')
    master_df = calcDerivedFeatures(master_df, 'curr')
    master_df.to_csv(Import_Files.latest_master_df,index = False)
    return(master_df)

def calcDerivedFeatures(master_df, prefix):
  master_df[prefix + '_cust_per_mi'] = (master_df['customers'] / master_df['oh_mile']).fillna(0)
  master_df[prefix + '_trims_per_mi'] = (master_df[prefix + '_no_of_trims'] / master_df['oh_mile']).fillna(0)
  master_df[prefix + '_rem_per_mi'] = (master_df[prefix + '_no_of_rem'] / master_df['oh_mile']).fillna(0)
  master_df[prefix + '_brush_per_mi'] = (master_df[prefix + '_brush_acres'] / master_df['oh_mile']).fillna(0)
  master_df[prefix + '_tot_hr_per_mi'] = ((master_df[prefix + '_brush_hr'] +
                                   master_df[prefix + '_trim_hr'] +
                                   master_df[prefix + '_rem_hr'])/ master_df['oh_mile']).fillna(0)
  master_df[prefix + '_trim_hr_per_mi'] = (master_df[prefix + '_trim_hr'] / master_df['oh_mile']).fillna(0)
  master_df[prefix + '_rem_hr_per_mi'] = (master_df[prefix + '_rem_hr'] / master_df['oh_mile']).fillna(0)
  master_df[prefix + '_brush_hr_per_mi'] = (master_df[prefix + '_brush_hr'] / master_df['oh_mile']).fillna(0)
  master_df[prefix + '_trims_per_hr'] = (master_df[prefix + '_no_of_trims'] / master_df[prefix + '_trim_hr']).fillna(0)
  master_df[prefix + '_rem_per_hr'] = (master_df[prefix + '_no_of_rem'] / master_df[prefix + '_rem_hr']).fillna(0)
  master_df[prefix + '_brush_per_hr'] = (master_df[prefix + '_brush_acres'] / master_df[prefix + '_brush_hr']).fillna(0)
  return master_df

def preProcessData(master_df,target_feature,scale ,drop_feat_list):
    if scale == 0:
        master_df = master_df.drop (drop_feat_list,1,errors='ignore')
        y = master_df[target_feature]
        X = master_df.drop([target_feature],1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8)
        train_circuits , test_circuits = X_train.circuit_id , X_test.circuit_id
        X_train , X_test = X_train.drop(['circuit_id'],1),X_test.drop(['circuit_id'],1)
        return(X_train, X_test, y_train, y_test,train_circuits,test_circuits)
    if scale == 1:
        master_df = master_df.drop(drop_feat_list,1,errors='ignore')
        master_df.index = master_df.circuit_id
        master_df = master_df.drop('circuit_id',1)
        train_circuits, test_circuits =  train_test_split(master_df.index)
        y = master_df[target_feature]
        X = master_df.drop(target_feature,1)
        y_train_unscld,y_test_unscld = y[y.index.isin(train_circuits)],y[y.index.isin(test_circuits)]
        X_train_unscld, X_test_unscld = X[X.index.isin(train_circuits)],X[X.index.isin(test_circuits)]
        scaler_X = StandardScaler().fit(X_train_unscld)
        scaler_y = StandardScaler().fit(y_train_unscld.values.reshape(-1,1))
        X_train_scld = pd.DataFrame(scaler_X.transform(X_train_unscld),columns = X_train_unscld.columns,index =X_train_unscld.index )
        X_test_scld = pd.DataFrame(scaler_X.transform(X_test_unscld),columns = X_test_unscld.columns,index = X_test_unscld.index)
        y_train_scld = pd.DataFrame(scaler_y.transform(y_train_unscld.values.reshape(-1,1)),columns = [y_train_unscld.name], index = y_train_unscld.index)
        y_test_scld = pd.DataFrame(scaler_y.transform(y_test_unscld.values.reshape(-1,1)),columns = [y_test_unscld.name], index = y_test_unscld.index)
        return(X_train_unscld, X_test_unscld,y_train_unscld,y_test_unscld,
               scaler_X,scaler_y,X_train_scld,X_test_scld,y_train_scld,y_test_scld)

def buildRandomForestRegCV(X_train,y_train):
    ### Random forest regression
    model = RandomForestRegressor()
    param_dist = {
              'n_estimators': (1000,250),
              "max_depth": [10,100,1000],
              "max_features": [3,4,7,9,12],
              "min_samples_split": [2,3],
              "min_samples_leaf": [2,3],
              "bootstrap": [True, False]}
    grid_search = RandomizedSearchCV(model,
                               param_distributions = param_dist,
                     n_jobs = 1,verbose = 1)
    grid_search.fit(X_train,y_train)
    return(grid_search)

def buildSVMCV(X_train_scaled,y_train_scaled):
    pipeSVR = Pipeline([('scaler', StandardScaler()),('model',SVR())])
    parameters = [{'model__C':[0.01,1,10,100,1000],
                   'model__kernel':['linear']},
                  {'model__C':[0.01,0.1,1,10,100,1000],
                   'model__kernel':['rbf'],
                   'model__gamma':[0.1,0.01,0.001,0.0001]}]
    '''
                 {'model__C':[0.01,0.1,1,10,100,1000],
                   'model__kernel':['poly'],
                   'model__degree':[2,3,4]}]'''
    grid_search = GridSearchCV(estimator = pipeSVR,param_grid = parameters,
                     scoring = 'r2', n_jobs = -1, cv  =5,verbose = 1)
    grid_search.fit(X_train_scaled,y_train_scaled)
    return(grid_search)

def buildElasticNetCV(X_train_scaled,y_train_scaled):
    pipeEN = Pipeline([('scaler', StandardScaler()),('model',ElasticNet())])
    parameters = [{'model__alpha':[0.1,0.2,0.4,0.3,0.6,0.5,0.7,0.8,0.9,1.0]}]
    grid_search = GridSearchCV(estimator = pipeEN,param_grid = parameters,
                                n_jobs = -1, cv  = 10,verbose = 1)
    grid_search.fit(X_train_scaled,y_train_scaled)
    return(grid_search)

def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

def featureSimilarityEngine(X_train_scld,scaler_X, test_X_sample_unscld,
                            y_train_unscld,num_comparisons,target_feature):
    X_test_sampled_scaled = scaler_X.transform(test_X_sample_unscld)
    df = pd.DataFrame(sklearn.metrics.pairwise.pairwise_distances(X_test_sampled_scaled,
                      Y=X_train_scld,metric='euclidean', n_jobs=1),index = ['distance']
                       ,columns = X_train_scld.index).T
    df[target_feature] = y_train_unscld
    df = df.sort_values('distance')
    df = df.ix[range(0, num_comparisons)]
    return(df)

'''
def ensembleBestFits(scaled_data_models_list,unscaled_data_models_list,
                     weighted = 0,X_test_scaled,y_test_scaled,
                     X_test_unscaled,y_test_unscaled,scaler_y):
    scaled_predictions = []
    for m in scaled_data_models_list:
        scaled_predictions.append(m.best_estimator_.predict(X_test_scaled))

def costSimilarityEngine(master_df, target_feature,predicted_feature, range_percent = 2):
    range_percent = range_percent/100
    feature_range = range(round(predicted_feature * (1 - range_percent)),round(predicted_feature * (1 + range_percent)))
    similar_trims = master_df[(master_df[target_feature] > min(cost_range))]
    similar_trims = similar_trims[(similar_trims[target_feature] < max(cost_range))]
    return(similar_trims)
'''