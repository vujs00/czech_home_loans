# -*- coding: utf-8 -*-
"""
"""


#------------------------------------------------------------#
# STEP 1: general imports and paths                          #
#------------------------------------------------------------#

# Data manipulation.
import pandas as pd

# Save model.
from joblib import dump, load

# Load dataset.
df0 = pd.read_csv(r'C:\Users\JF13832\Downloads\Thesis\02 Data\01 Source\czech_mortgages_dataset_v2.csv', 
                  decimal = ',',
                  delimiter = '|',
                  encoding = 'cp437')

# Variable name mapping dataset.
var_map0 = pd.read_excel(r'C:\Users\JF13832\Downloads\Thesis\02 Data\01 Source\mapping_tables.xlsx',
                         sheet_name = 'variables')

#------------------------------------------------------------#
# STEP 2: run                                                #
#------------------------------------------------------------#

from DataGetter import DataGetter
from Preprocessor import Preprocessor
from Modeler import Modeler
from Validator import Validator

from sklearn.linear_model import LogisticRegression

def run(df_in, var_map_in, set_seed, target, undersample, decorrelate, oot_year, 
        encoding_metric, selection_metric, select_features_bool):
    
    df, var_map = DataGetter(df_in, var_map_in).run()
    
    df_train, df_test, df_oot, performance_summary =\
        Preprocessor(set_seed, target, df, undersample, decorrelate,
                     oot_year, encoding_metric).run()
    
    (logit, ann, knn, svm, bag, rf, adaboost,
     X_train, X_test, X_oot, y_train, y_test, y_oot) =\
        Modeler(set_seed, target, decorrelate, df_train,
                df_test, df_oot).run(selection_metric, select_features_bool,
                                     encoding_metric)
    
    aucs = Validator(X_train, y_train, X_test, y_test,
                     X_oot, y_oot).run([logit, ann, knn, svm,
                                        bag, rf, adaboost])

    return logit, ann, knn, svm, bag, rf, adaboost, aucs


# Single run.
(logit, ann, knn, svm,
 bag, rf, adaboost, aucs) = run(df0, var_map0, 130816,
                                'default_event_flg', False, False, 201901,
                                'bins', LogisticRegression(), False)

# Dumps.
dump(logit, r'C:\Users\JF13832\Downloads\Thesis\03 Models\3_logit.joblib')
dump(ann, r'C:\Users\JF13832\Downloads\Thesis\03 Models\3_ann.joblib')
dump(knn, r'C:\Users\JF13832\Downloads\Thesis\03 Models\3_knn.joblib')
dump(svm, r'C:\Users\JF13832\Downloads\Thesis\03 Models\3_svm.joblib')
dump(bag, r'C:\Users\JF13832\Downloads\Thesis\03 Models\3_bag.joblib')
dump(rf, r'C:\Users\JF13832\Downloads\Thesis\03 Models\3_rf.joblib')
dump(adaboost, r'C:\Users\JF13832\Downloads\Thesis\03 Models\3_adaboost.joblib')

# Params.
logit.best_params_
ann.best_params_
knn.best_params_
svm.best_params_
bag.best_params_
rf.best_params_
adaboost.best_params_

