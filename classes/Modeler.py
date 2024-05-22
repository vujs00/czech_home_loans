# -*- coding: utf-8 -*-
"""
"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


class Modeler():
    
    
    def __init__(self, set_seed, target, decorrelate, 
                 df_train, df_test, df_oot):
        self.set_seed: int = set_seed
        self.target: str = target
        self.decorrelate: bool = decorrelate
        self.df_train: pd.DataFrame = df_train
        self.df_test: pd.DataFrame = df_test
        self.df_oot: pd.DataFrame = df_oot
    
    
    def select_features(self, selection_metric, select_features_bool):
        
        if select_features_bool:
                    
            # Define algorithms
            sfs = SequentialFeatureSelector(selection_metric,
                                            #n_features_to_select = 15,
                                            scoring = 'roc_auc',
                                            direction = 'forward',
                                            cv = 5)
    
            # Define y on all samples.
            self.y_test = self.df_test.loc[:, self.target]
            self.y_train = self.df_train.loc[:, self.target]
            self.y_oot = self.df_oot.loc[:, self.target]
            
            # Define X for training sample.
            self.X_train = self.df_train.loc[:, self.df_train.columns!=self.target]
            
            # Fit to train.
            sfs.fit(self.X_train, self.y_train)
    
            # Update X matrices of all samples..
            #self.X_train = sfs.transform(self.X_train)
            self.X_train = self.df_train[list(sfs.get_feature_names_out())]
            self.X_test = self.df_test[list(sfs.get_feature_names_out())]
            self.X_oot = self.df_oot[list(sfs.get_feature_names_out())]
            
            print(list(sfs.get_feature_names_out()))
            
        else:
            
                # Define y on all samples.
                self.y_test = self.df_test.loc[:, self.target]
                self.y_train = self.df_train.loc[:, self.target]
                self.y_oot = self.df_oot.loc[:, self.target]
    
                # Define X on all samples.            
                self.X_train = self.df_train.loc[:, self.df_train.columns!=self.target]
                self.X_test = self.df_test.loc[:, self.df_test.columns!=self.target]
                self.X_oot = self.df_oot.loc[:, self.df_oot.columns!=self.target]
            
        
        return (self.X_train, self.y_train, self.X_test, self.y_test,
                self.X_oot, self.y_oot)


    def apply_one_hot(self, encoding_metric):
        
        if encoding_metric == 'bins':

            ### Categorize.
            self.X_train = self.X_train.astype('category')
            self.X_test = self.X_test.astype('category')
            self.X_oot = self.X_oot.astype('category')

            ### Apply one-hot.
            self.X_train = pd.get_dummies(self.X_train)
            self.X_test = pd.get_dummies(self.X_test)
            self.X_oot = pd.get_dummies(self.X_oot)

            # Keep only the interseciton of variables.
            inter = set(self.X_train).intersection(self.X_test, self.X_oot)
            
            self.X_train = self.X_train[list(inter)]
            self.X_test = self.X_test[list(inter)]
            self.X_oot = self.X_oot[list(inter)]
        
        else:
            pass
        
        return (self.X_train, self.X_test, self.X_oot)  
    
    def model_logit(self):
        
        if self.decorrelate:

            self.logit = LogisticRegression(solver = 'saga', max_iter = 1000,
                                            penalty = None)
            
            self.logit.fit(self.X_train, self.y_train)
    
            #SM = sm.Logit(self.y_train, self.X_train).fit()
            #print(SM.summary())
        
        else:

            self.logit = LogisticRegression(solver = 'saga', max_iter = 1000,
                                            penalty = 'elasticnet')
            
            hyperparameter_grid = {
                'l1_ratio' : [(0), (0.1), (0.2), (0.3), (0.4), (0.5),
                              (0.6), (0.7), (0.8), (0.9), (1)]
                }
            
            self.logit = GridSearchCV(self.logit, hyperparameter_grid, 
                                      n_jobs = -1, cv = 5)

            self.logit.fit(self.X_train, self.y_train)

        return self.logit
    
    
    def model_ann(self):
        
        mlp = MLPClassifier(max_iter = 1000, random_state = self.set_seed,
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
        
        self.ann = GridSearchCV(mlp, hyperparameter_grid, n_jobs = -1, cv = 5)
        
        self.ann.fit(self.X_train, self.y_train)
        
        print('Best parameters found:\n', self.ann.best_params_)
        
        for mean, std, params in zip(self.ann.cv_results_['mean_test_score'],
                                     self.ann.cv_results_['std_test_score'],
                                     self.ann.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))    
            
        return self.ann
    
 
    def model_knn(self):

        hyperparameter_grid = {
            'n_neighbors' : [1, 5, 10, 30, 50, 100],
            'weights' : ('uniform', 'distance'),
            'p' : [(1), (2)]
            }
        
        self.knn = GridSearchCV(KNeighborsClassifier(), hyperparameter_grid, 
                                n_jobs = -1, cv = 5)
        self.knn.fit(self.X_train, self.y_train)

        print('Best parameters found:\n', self.knn.best_params_)
        
        for mean, std, params in zip(self.knn.cv_results_['mean_test_score'],
                                     self.knn.cv_results_['std_test_score'],
                                     self.knn.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
        return self.knn


    def model_svm(self):    

        svc = SVC(max_iter = 1000, probability = True, cache_size = 5000)
    
        hyperparameter_grid = {
            'C' : [(0), (0.25), (0.5), (0.75), (1)],
            'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
            'degree' : [(1), (2), (3)]
            }
        
        self.svm = GridSearchCV(svc, hyperparameter_grid, n_jobs = -1, cv = 5)
        
        self.svm.fit(self.X_train, self.y_train)
     
        print('Best parameters found:\n', self.svm.best_params_)
        
        for mean, std, params in zip(self.svm.cv_results_['mean_test_score'],
                                     self.svm.cv_results_['std_test_score'],
                                     self.svm.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))   
                
        return self.svm


    def model_bagging(self):
        
        hyperparameter_grid = {
            'n_estimators': [10, 20, 30, 50, 100, 150]
            }
        
        self.bag = GridSearchCV(BaggingClassifier(random_state = self.set_seed), 
                                param_grid = hyperparameter_grid,
                                cv = 5, n_jobs = -1)
        self.bag.fit(self.X_train, self.y_train)
            
        for mean, std, params in zip(self.bag.cv_results_['mean_test_score'],
                                     self.bag.cv_results_['std_test_score'],
                                     self.bag.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))   
        
        return self.bag


    def model_rf(self):
        
        hyperparameter_grid = {
            'bootstrap' : [True],
            'max_depth' : [1, 2, 3, 4, 5, 10, 15],
            'min_samples_leaf' : [10],
            'n_estimators': [10, 20, 30, 50, 100, 150]
            }
    
        self.rf = GridSearchCV(RandomForestClassifier(random_state = self.set_seed),
                               param_grid = hyperparameter_grid, cv = 5, 
                               n_jobs = -1)
        self.rf.fit(self.X_train, self.y_train)

        for mean, std, params in zip(self.rf.cv_results_['mean_test_score'],
                                     self.rf.cv_results_['std_test_score'],
                                     self.rf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))   

        return self.rf

    
    def model_adaboost(self):
        
        hyperparameter_grid = {
            'n_estimators': [10, 20, 30, 50, 100, 150]
            }
        
        self.adaboost = GridSearchCV(AdaBoostClassifier(random_state = self.set_seed),
                                     param_grid = hyperparameter_grid, cv = 5,
                                     n_jobs = -1)
        self.adaboost.fit(self.X_train, self.y_train)

        for mean, std, params in zip(self.adaboost.cv_results_['mean_test_score'],
                                     self.adaboost.cv_results_['std_test_score'],
                                     self.adaboost.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))           
        
        return self.adaboost


    def run(self, selection_metric, select_features_bool, encoding_metric):
        
        self.select_features(selection_metric, select_features_bool)
        self.apply_one_hot(encoding_metric)
        self.model_logit()
        self.model_ann()
        self.model_knn()
        self.model_svm()
        self.model_bagging()
        self.model_rf()
        self.model_adaboost()
        
        return (self.logit, self.ann, self.knn,  self.svm,
                self.bag, self.rf, self.adaboost,
                self.X_train, self.X_test, self.X_oot, 
                self.y_train, self.y_test, self.y_oot)
        
    
    