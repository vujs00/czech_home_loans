# -*- coding: utf-8 -*-
"""
"""


#------------------------------------------------------------#
# STEP 1: general imports and paths                          #
#------------------------------------------------------------#

# Data manipulation.
import pandas as pd

# Load dataset.
df0 = pd.read_csv(r'C:\Users\JF13832\Downloads\Thesis\03 Data\01 Source\czech_mortgages_dataset_v2.csv', 
                  decimal = ',',
                  delimiter = '|',
                  encoding = 'cp437')

helper_1 = df0.loc[df0['DefaultEvent'] == 1]
helper_2 = df0.loc[df0['DefaultEvent'] == 0]
helper_2 = helper_2.head(5000)
df0 = pd.concat([helper_1, helper_2])

# Variable name mapping dataset.
var_map0 = pd.read_excel(r'C:\Users\JF13832\Downloads\Thesis\03 Data\01 Source\mapping_tables.xlsx',
                         sheet_name = 'variables')

#------------------------------------------------------------#
# STEP 2: run                                                #
#------------------------------------------------------------#

from Parent import Parent
from DataGetter import DataGetter
from Preprocessor import Preprocessor
from Modeler import Modeler
from Validator import Validator

from sklearn.linear_model import LogisticRegression

def run(df_in, var_map_in, set_seed, target, encoding_metric, selection_metric):
    
    df, var_map = DataGetter(df_in, var_map_in).run()
    df_train, df_test = Preprocessor(set_seed, target, df, encoding_metric).run()
    (logit, ann, knn, svm, rf,
     X_train, X_test, y_train, y_test) =\
        Modeler(set_seed, target, df_train, df_test).run(selection_metric)
    Validator(X_train, y_train, X_test, y_test).run(logit)
    Validator(X_train, y_train, X_test, y_test).run(ann)
    Validator(X_train, y_train, X_test, y_test).run(knn)
    Validator(X_train, y_train, X_test, y_test).run(svm)
    Validator(X_train, y_train, X_test, y_test).run(rf)

    return logit, ann, knn, svm, rf
    
logit, ann, knn, svm, rf = run(df0, var_map0, 123, 'default_event_flg',
                               'woe', LogisticRegression())




