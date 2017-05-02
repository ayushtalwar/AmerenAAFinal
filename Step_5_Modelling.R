#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library(data.table)
library(bit64)
path_to_data = '/Users/Ayush Talwar/Documents/Ameren/Data/'
# X_train_unscaled = fread(paste(path_to_data , 'X_train_unscaled.csv',sep = ''))
# y_train_unscaled = fread(paste(path_to_data , 'y_train_unscaled.csv',sep = ''))
# X_test_unscaled = fread(paste(path_to_data , 'X_test_unscaled.csv',sep = ''))
# y_test_unscaled = fread(paste(path_to_data , 'y_test_unscaled.csv',sep = ''))


X_train_unscaled = fread(args[1])
y_train_unscaled = fread(args[2])
X_test_unscaled = fread(args[3])
y_test_unscaled = fread(args[4])


train_data = cbind(X_train_unscaled,y_train_unscaled)
ctrl = trainControl(method = 'cv', number = 5)
xgbGrid <- expand.grid(nrounds = c(256,16),max_depth = c(1,2), eta = c(0.04,0.08),
                       gamma = 0,colsample_bytree = c( 0.7),min_child_weight = c(2,4))
ctrl <- gafsControl(functions = caretGA)

library(caret)
xg_model = train(colnames(y_train_unscaled)[1]
                 ~., train_data, tuneGrid =xgbGrid,
                 gafsControl = ctrl, method = 'xgbTree',iters = 1000000)
v = varImp(xg_model)
write.csv(v$importance,file = args[5])
save(xg_model, file = args[6])

import subprocess
import sys
import argparse
import os
import subprocess
parser = argparse.ArgumentParser( prog='parser.py', 
                                  usage='%(prog)s -m [file.txt]')
parser.add_argument('-m', required=True, help=path_to_data + 'X_train_unscaled.csv', type=argparse.FileType('r'))
parser.add_argument('-o', required=True, help='Save output to file', type=argparse.FileType('w'))

args = parser.parse_args()

command = 'Rscript'
path2script = '/Users/Ayush Talwar/Documents/AmerenAA/Step_5_Modelling.R'
args =[ 
  "path_to_data + 'X_train_unscaled.csv'" , 
  "path_to_data + 'y_train_unscaled.csv'", 
  "path_to_data + 'X_test_unscaled.csv'" , 
  "path_to_data + 'y_test_unscaled.csv'",
  "'/Users/Ayush Talwar/Documents/Ameren/xg_model_importance.csv'",
  "'/Users/Ayush Talwar/Documents/Ameren/xg_model.rda'"]
cmd = [command, path2script] + args
subprocess.check_call(cmd, universal_newlines=True)


subprocess.call(["Rscript", path2script, "--args", args])


X_train_unscaled.to_csv(path_to_data + 'X_train_unscaled.csv',index = False)
pd.DataFrame(y_train_unscaled).to_csv(path_to_data + 'y_train_unscaled.csv',index = False)
X_test_unscaled.to_csv(path_to_data + 'X_test_unscaled.csv',index = False)
pd.DataFrame(y_test_unscaled).to_csv(path_to_data + 'y_test_unscaled.csv',index = False)

