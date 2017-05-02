#!/usr/bin/python
import pandas as pd
import Import_Files
import csv
class CircuitCharacteristicsFactory():
    def __init__(self,circuit_characteristic_file,output_file_path):
        self.circuit_characteristic_file = circuit_characteristic_file
        self.circuit_characteristics_select = pd.DataFrame()
        self.output_file_path = output_file_path
        self.loadRaw()

    def loadRaw(self):
        sql_flg = 0
        with open(Import_Files.sql_flag) as f:
            csv_f = csv.reader(f)
            for row in csv_f:
                if row[0] == '1':
                    sql_flg = 1
                 
        if sql_flg == 1:
            circuit_characteristics = pd.read_sql(Import_Files.circuit_characteristics_table,Import_Files.engine)
        else:    
            circuit_characteristics = pd.read_excel(self.circuit_characteristic_file)        
        
        circuit_characteristics.columns = map(str.lower, circuit_characteristics.columns)
        circuit_characteristics.columns = circuit_characteristics.columns.str.replace ('[^0-9A-Za-z]','_')
        circuit_characteristics = circuit_characteristics[circuit_characteristics['feeder']  != 'DEAD']
        circuit_characteristics['substation_number'] = circuit_characteristics.feeder.str.slice(0,3,1)
        circuit_characteristics.feeder = circuit_characteristics.feeder.str.slice(3,6,1)
        self.circuit_characteristics_select = circuit_characteristics[['src_op_center','feeder','substation_number',
                                                                       '__customers','oh_mile','ug_mile',
                                                                       'src_volts_class']].rename(columns={'__customers': 'customers',
                                                                                                           'src_volts_class': 'voltage'})
        self.circuit_characteristics_select['voltage'] = self.circuit_characteristics_select['voltage'].str.replace('[^0-9]','')
        self.circuit_characteristics_select['sub_fed'] = self.circuit_characteristics_select['substation_number'] + self.circuit_characteristics_select['feeder']
        self.circuit_characteristics_select = self.circuit_characteristics_select.drop(['substation_number','feeder'],1)

    def getFeatures(self):
         return(self.circuit_characteristics_select)

    def writeFeatures(self):
        self.circuit_characteristics_select.to_pickle(self.output_file_path)
