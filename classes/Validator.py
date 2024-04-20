# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:00:34 2024

@author: JF13832
"""


#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

from dataclasses import dataclass 

import pandas as pd
import numpy as np

#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#

@dataclass(frozen = False)
class Validator:
    
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    
    
    def predict(self, model):
        
        self.y_train_pred = model.predict_proba(self.X_train)[:,1]
        self.y_test_pred = model.predict_proba(self.X_test)[:,1]
    
        return (self.y_train_pred, self.y_test_pred)
    
    
    def plot_roc(self):
        
        plt.figure(figsize=(5, 5))
         
        fpr_train, tpr_train, _ = roc_curve(self.y_train, self.y_train_pred)
        roc_auc_train = auc(fpr_train, tpr_train)
        
        fpr_test, tpr_test, _ = roc_curve(self.y_test, self.y_test_pred)
        roc_auc_test = auc(fpr_test, tpr_test)
          
        plt.plot(fpr_train, tpr_train, ':',
                 label = f'(train set, AUC = {roc_auc_train:.2f})', color = 'k')
        plt.plot(fpr_test, tpr_test,
                 label = f'(test set, AUC = {roc_auc_test:.2f})', color = 'k')
        plt.plot([0, 1], [0, 1], '--', color = 'k')
         
        # Labels
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        for pos in ['right', 'top']: 
            plt.gca().spines[pos].set_visible(False) 
        plt.legend()
        plt.legend()
        plt.show()
    
    
    def plot_cap(self):
            
        
        def prep_cap_inputs(y, y_pred):
            
            N = len(y)
            positives = np.sum(y)
            y_sort = y[y_pred.argsort()[::-1][:N]]
            p_positives = np.append([0], np.cumsum(y_sort))/positives
            p_N = np.arange(0, N + 1) / N
            fpr, tpr, _ = roc_curve(y, y_pred)
            roc_auc = auc(fpr, tpr)
            gini = roc_auc * 2 - 1
                    
            return N, positives, p_N, p_positives, gini
    
    
        # Initialization.
        plt.figure(figsize=(5, 5))
    
        # Random model.
        plt.plot([0, 1], [0, 1], '--', color = 'k')
    
        # Train.
        N, positives, p_N, p_positives, gini =\
            prep_cap_inputs(self.y_train, self.y_train_pred)
        plt.plot(p_N, p_positives, ":", color="k",
                 label="train set, Gini: {:.5f}".format(gini))
    
        # Test.
        N, positives, p_N, p_positives, gini =\
            prep_cap_inputs(self.y_test, self.y_test_pred)
        plt.plot(p_N, p_positives, color="k",
                 label="test set, Gini: {:.5f}".format(gini))    
    
        # Perfect model plot.
        plt.plot([0, positives / N, 1], [0, 1, 1], color='k',
                 linestyle='--', label="Perfect model")
        
        # Labels.
        plt.xlabel('Fraction of all population')
        plt.ylabel('Fraction of event population')
        for pos in ['right', 'top']: 
            plt.gca().spines[pos].set_visible(False) 
        plt.legend(loc='lower right')
    
    
    def plot_ks(self):
    
        
        def prep_ks_inputs(y, y_pred):
            
            N = len(y)
            positives = np.sum(y)
            negatives = N - positives
            ids = y_pred.argsort()
            y_sort = y[ids]
            y_pred_sort = y_pred[ids]
            
            N_cum = np.arange(0, N)
            positives_cum = np.cumsum(y_sort)
            negatives_cum = N_cum - positives_cum
        
            p_positives = positives_cum / positives
            p_negatives = negatives_cum / negatives
        
            p_diff = p_positives - p_negatives
            ks_score = np.max(p_diff)
            ks_max_id = np.argmax(p_diff)
                    
            return y_pred_sort, p_positives, p_negatives, ks_max_id, ks_score
    
    
        # Initialization.
        plt.figure(figsize=(5, 5))
    
        # Train set.
        y_pred_sort, p_positives, p_negatives, ks_max_id, ks_score =\
            prep_ks_inputs(self.y_train, self.y_train_pred)
        
        plt.plot(y_pred_sort, p_positives, ":", color="k", 
                 label="train set defaults")
        plt.plot(y_pred_sort, p_negatives, ":", color="k", 
                 label="train set non-defaults")
    
        # Test set.
        y_pred_sort, p_positives, p_negatives, ks_max_id, ks_score =\
            prep_ks_inputs(self.y_test, self.y_test_pred)
        
        plt.plot(y_pred_sort, p_positives, color="k",
                 label="test set defaults")
        plt.plot(y_pred_sort, p_negatives, color="k", 
                 label="test set non-defaults")
         
        plt.vlines(y_pred_sort[ks_max_id], ymin=p_positives[ks_max_id],
                   ymax=p_negatives[ks_max_id], color="k", linestyles="--")
    
        pos_x = p_positives[ks_max_id] + 0.02
        pos_y = 0.5 * (p_negatives[ks_max_id] + p_positives[ks_max_id])
        text = "KS: {:.2%} at {:.2f}".format(ks_score, p_positives[ks_max_id])
        plt.text(pos_x, pos_y, text, fontsize=12, rotation_mode="anchor")
        for pos in ['right', 'top']: 
            plt.gca().spines[pos].set_visible(False) 
        plt.legend(loc='lower right')
    

    def run(self, model):
        
        self.predict(model)
        self.plot_roc()
        self.plot_cap()
        self.plot_ks()
        
        
