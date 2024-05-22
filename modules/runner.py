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
                                'default_event_flg', False, True, 201901,
                                'bins', LogisticRegression(), True)

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


###############################################################################
###############################################################################                                
###############################################################################                                
###############################################################################
###############################################################################


# Definitions for single runs.
df_in = df0.copy(deep = True)
var_map_in = var_map0.copy(deep = True)
set_seed = 130816
target = 'default_event_flg'
undersample = False
oot_year = 201901
decorrelate = True
encoding_metric = 'bins'
selection_metric = LogisticRegression()
select_features_bool = True

# Single runs.

df, var_map = DataGetter(df_in, var_map_in).run()

df_train, df_test, df_oot, performance_summary =\
    Preprocessor(set_seed, target, df, undersample, decorrelate,
                 oot_year, encoding_metric).run()

(X_train, y_train, X_test, y_test,
 X_oot, y_oot) =\
    Modeler(set_seed, target, decorrelate, df_train,
            df_test, df_oot).select_features(selection_metric, select_features_bool)


def apply_one_hot(X_train, X_test, X_oot):
    
        ### Categorize.
        X_train = X_train.astype('category')
        X_test = X_test.astype('category')
        X_oot = X_oot.astype('category')

        ### Apply one-hot.
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)
        X_oot = pd.get_dummies(X_oot)

        # Keep only the interseciton of variables.
        inter = set(X_train).intersection(X_test, X_oot)
        
        X_train = X_train[list(inter)]
        X_test = X_test[list(inter)]
        X_oot = X_oot[list(inter)]
        
        return (X_train, X_test, X_oot)  

X_train,X_test,X_oot = apply_one_hot(X_train, X_test, X_oot)
#a.T.apply(lambda x: x.nunique(), axis=1)



from sklearn.model_selection import GridSearchCV

#hyperparameter_grid = {
#    'l1_ratio' : [(0), (0.1), (0.2), (0.3), (0.4), (0.5),
#                  (0.6), (0.7), (0.8), (0.9), (1)]
#    }

#logit = LogisticRegression(solver = 'saga', max_iter = 1000,
#                                penalty = 'elasticnet')
#logit = GridSearchCV(logit, hyperparameter_grid, 
#                          n_jobs = -1, cv = 5)

#logit.fit(X_train, y_train)

#logit.best_params_
#aucs = Validator(X_train, y_train, X_test, y_test,
#                 X_oot, y_oot).run([logit])


logit = LogisticRegression(solver = 'saga', max_iter = 1000,
                           penalty = None)
logit.fit(X_train, y_train)

aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([logit])

import statsmodels.api as sm
SM = sm.Logit(y_train, X_train).fit(maxiter = 1000)
print(SM.summary())
sm_summary = SM.summary()
print(sm_summary.as_csv())

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter = 1000, random_state = set_seed,
                    early_stopping = True)

hyperparameter_grid = {
    'hidden_layer_sizes': [(7),
                           (5, 5),
                           (7, 7),
                           (10, 5),
                           (10, 10),
                           (20, 20),
                           (40, 40, 20),
                           (40, 20, 20),
                           (20, 40, 20),
                           (100, 200, 100),
                           (200, 100, 100),
                           (200, 200, 100),
                           (200, 400, 200)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

ann = GridSearchCV(mlp, hyperparameter_grid, n_jobs = -1, cv = 5)

ann.fit(X_train, y_train)
aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([ann])


hyperparameter_grid = {
    'n_neighbors' : [1, 5, 10, 30, 50, 100],
    'weights' : ('uniform', 'distance'),
    'p' : [(1), (2)]
    }
from sklearn.neighbors import KNeighborsClassifier

knn = GridSearchCV(KNeighborsClassifier(), hyperparameter_grid, 
                   n_jobs = -1, cv = 5)
knn.fit(X_train, y_train)
knn.best_params_
aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([knn])




from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC(max_iter = 1000, probability = True, cache_size = 5000)

hyperparameter_grid = {
    'C' : [(0), (0.25), (0.5), (0.75), (1)],
    'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
    'degree' : [(1), (2), (3)]
    }

svm = GridSearchCV(svc, hyperparameter_grid, n_jobs = -1, cv = 5)

svm.fit(X_train, y_train)
 
svm.best_params_


aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([svm])






from sklearn.ensemble import BaggingClassifier

hyperparameter_grid = {
    'n_estimators': [10, 20, 30, 50, 100, 150]
    }

bag = GridSearchCV(BaggingClassifier(random_state = set_seed), 
                   param_grid = hyperparameter_grid,
                   cv = 5, n_jobs = -1)
bag.fit(X_train, y_train)
bag.best_params_
aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([bag])
    



hyperparameter_grid = {
    'bootstrap' : [True],
    'max_depth' : [1, 2, 3, 4, 5, 10, 15],
    'min_samples_leaf' : [10],
    'n_estimators': [10, 20, 30, 50, 100, 150]
    }


from sklearn.ensemble import RandomForestClassifier

rf = GridSearchCV(RandomForestClassifier(random_state = set_seed),
                  param_grid = hyperparameter_grid, cv = 5, 
                  n_jobs = -1)
rf.fit(X_train, y_train)
rf.best_params_
aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([rf])




hyperparameter_grid = {
    'n_estimators': [10, 20, 30, 50, 100, 150]
    }

from sklearn.ensemble import AdaBoostClassifier

adaboost = GridSearchCV(AdaBoostClassifier(random_state = set_seed),
                             param_grid = hyperparameter_grid, cv = 5,
                             n_jobs = -1)
adaboost.fit(X_train, y_train)

adaboost.best_params_

aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([adaboost])


# Loop run.
for oot_i in [201201, 201301, 201401, 201501, 201601, 201701, 201801, 201901]:
    (logit, ann, knn, 
     bag, rf, adaboost, aucs) = run(df0, var_map0, 130816,
                                    'default_event_flg', False, oot_i,
                                    'woe', LogisticRegression())





logit = load(r'C:\Users\JF13832\Downloads\Thesis\03 Models\1_logit.joblib')
logit.best_params_
aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([logit])

ann = load(r'C:\Users\JF13832\Downloads\Thesis\03 Models\1_ann.joblib')
ann.best_params_
aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([ann])

knn = load(r'C:\Users\JF13832\Downloads\Thesis\03 Models\1_knn.joblib')
knn.best_params_
aucs = Validator(X_train, y_train, X_test, y_test,
                 X_oot, y_oot).run([knn])

#svm.best_params_

bag = load(r'C:\Users\JF13832\Downloads\Thesis\03 Models\1_bag.joblib')
bag.best_params_

rf = load(r'C:\Users\JF13832\Downloads\Thesis\03 Models\1_rf.joblib')
rf.best_params_

adaboost = load(r'C:\Users\JF13832\Downloads\Thesis\03 Models\1_adaboost.joblib')
adaboost.best_params_

