!unzip "/content/hw3_part1_data.zip"

import os
import csv
import numpy as np

data = {}
data_path = "/content/all_data"
count = 0
for data_file in os.listdir(data_path):
  count+=1
  print(count, data_file)
  with open(data_path+"/"+data_file, "r", errors="ignore") as f:
    file_content = csv.reader(f, delimiter=',')
    data_set = []
    for line in file_content:
      data_set.append(list(map(int, line)))
  if "train" in data_file:
    data_set_name = data_file.replace("train_", "").replace(".csv", "")
    if data_set_name not in data:
      data[data_set_name] = {}
    data[data_set_name]["train"] = data_set
  elif "valid" in data_file:
    data_set_name = data_file.replace("valid_", "").replace(".csv", "")
    if data_set_name not in data:
      data[data_set_name] = {}
    data[data_set_name]["validation"] = data_set
  elif "test" in data_file:
    data_set_name = data_file.replace("test_", "").replace(".csv", "")
    if data_set_name not in data:
      data[data_set_name] = {}
    data[data_set_name]["test"] = data_set

#Splitting the features and the class
processed_data = {}
for data_set in data.keys():
  processed_data[data_set] = {}
  for data_set_type in data[data_set].keys():
    row_length = len(data[data_set][data_set_type][0])
    data[data_set][data_set_type] = np.array(data[data_set][data_set_type])
    processed_data[data_set][data_set_type + "_X"] = data[data_set][data_set_type][:,0:row_length - 1]
    processed_data[data_set][data_set_type + "_y"] = data[data_set][data_set_type][:,row_length - 1]

print(processed_data)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 

params = {"max_depth" : [6, 8, 10, 12, 14, 20, 25], "criterion" : [ "gini", "entropy"], "splitter" : ["best", "random"], "max_features" : ["auto", "sqrt", "log2"], "min_samples_split" : [2, 5, 10, 15], "min_samples_leaf" : [1, 2, 5, 10] }

for data_set in processed_data.keys():
  t_clf = GridSearchCV(DecisionTreeClassifier(), params, cv = 2)
  t_clf.fit(processed_data[data_set]["train_X"] + processed_data[data_set]["validation_X"], processed_data[data_set]["train_y"] + processed_data[data_set]["validation_y"])
  tuned_parameters = t_clf.best_params_
  print(data_set, t_clf.best_params_,end="\n")
  clf = DecisionTreeClassifier(max_depth = tuned_parameters["max_depth"], criterion = tuned_parameters["criterion"], splitter = tuned_parameters["splitter"], max_features=tuned_parameters["max_features"], min_samples_split= tuned_parameters["min_samples_split"], min_samples_leaf = tuned_parameters["min_samples_leaf"])
  clf.fit(processed_data[data_set]["train_X"] + processed_data[data_set]["validation_X"], processed_data[data_set]["train_y"] + processed_data[data_set]["validation_y"])
  processed_data[data_set]["test_y_pred"] = clf.predict(processed_data[data_set]["test_X"])
  print(metrics.classification_report(processed_data[data_set]["test_y"], processed_data[data_set]["test_y_pred"]))

from sklearn.ensemble import BaggingClassifier 

params = {"bootstrap" : [True, False], "bootstrap_features" : [True, False], "max_features" : [15, 20, 25, 30, 35]}

for data_set in processed_data.keys():
  t_clf = GridSearchCV(BaggingClassifier(), params, cv = 2)
  t_clf.fit(processed_data[data_set]["train_X"] + processed_data[data_set]["validation_X"], processed_data[data_set]["train_y"] + processed_data[data_set]["validation_y"])
  tuned_parameters = t_clf.best_params_
  print(data_set, tuned_parameters,end="\n")
  clf = BaggingClassifier(bootstrap=tuned_parameters["bootstrap"], bootstrap_features = tuned_parameters["bootstrap_features"], max_features = tuned_parameters["max_features"])
  clf.fit(processed_data[data_set]["train_X"] + processed_data[data_set]["validation_X"], processed_data[data_set]["train_y"] + processed_data[data_set]["validation_y"])
  processed_data[data_set]["test_y_pred"] = clf.predict(processed_data[data_set]["test_X"])
  print(metrics.classification_report(processed_data[data_set]["test_y"], processed_data[data_set]["test_y_pred"]))

from sklearn.ensemble import RandomForestClassifier 
params = {"max_depth" : [6, 8, 10, 12, 14, 20, 25], "criterion" : [ "gini", "entropy"], "max_features" : ["auto", "sqrt", "log2"], "min_samples_split" : [2, 5, 10, 15], "min_samples_leaf" : [1, 2, 5, 10] }

for data_set in processed_data.keys():
  t_clf = GridSearchCV(RandomForestClassifier(), params, cv = 2)
  t_clf.fit(processed_data[data_set]["train_X"] + processed_data[data_set]["validation_X"], processed_data[data_set]["train_y"] + processed_data[data_set]["validation_y"])
  tuned_parameters = t_clf.best_params_
  print(data_set, tuned_parameters,end="\n")
  clf = RandomForestClassifier(max_depth = tuned_parameters["max_depth"], criterion = tuned_parameters["criterion"], max_features=tuned_parameters["max_features"], min_samples_split= tuned_parameters["min_samples_split"], min_samples_leaf = tuned_parameters["min_samples_leaf"])
  clf.fit(processed_data[data_set]["train_X"] + processed_data[data_set]["validation_X"], processed_data[data_set]["train_y"] + processed_data[data_set]["validation_y"])
  processed_data[data_set]["test_y_pred"] = clf.predict(processed_data[data_set]["test_X"])
  print(metrics.classification_report(processed_data[data_set]["test_y"], processed_data[data_set]["test_y_pred"]))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

params = {"loss" : ["exponential", "deviance"], "criterion" : [ "friedman_mse", "squared_error"], "max_depth" : [3, 5, 7, 9], "min_samples_leaf" : [1, 2, 5, 10] }

temp = ['c1800_d5000', 'c1500_d100', 'c1800_d100', 'c500_d1000', 'c1000_d1000']

for data_set in temp:
  #data_set = "c500_d100"
  t_clf = GridSearchCV(GradientBoostingClassifier(), params, cv = 2, verbose=True, n_jobs=4)
  t_clf.fit(processed_data[data_set]["train_X"] + processed_data[data_set]["validation_X"], processed_data[data_set]["train_y"] + processed_data[data_set]["validation_y"])
  tuned_parameters = t_clf.best_params_
  print(data_set, tuned_parameters,end="\n")
  clf = GradientBoostingClassifier(loss = tuned_parameters["loss"], criterion = tuned_parameters["criterion"], max_depth= tuned_parameters["max_depth"], min_samples_leaf = tuned_parameters["min_samples_leaf"])
  clf.fit(processed_data[data_set]["train_X"] + processed_data[data_set]["validation_X"], processed_data[data_set]["train_y"] + processed_data[data_set]["validation_y"])
  processed_data[data_set]["test_y_pred"] = clf.predict(processed_data[data_set]["test_X"])
  print(metrics.classification_report(processed_data[data_set]["test_y"], processed_data[data_set]["test_y_pred"]))