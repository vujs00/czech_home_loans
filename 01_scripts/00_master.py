# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic
"""

#------------------------------------------------------------#
# STEP 1: libraries                                          #
#------------------------------------------------------------#

# Data management.
import numpy as np
import pandas as pd
import datetime

# Plots.
import matplotlib.pyplot as plt

# Preprocessing.
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from optbinning import BinningProcess
from collinearity import SelectNonCollinear
from sklearn.feature_selection import f_classif

# Modeling.
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

#------------------------------------------------------------#
# STEP 2: imports and paths                                  #
#------------------------------------------------------------#

# Load dataset.
df = pd.read_csv(r'C:\Users\JF13832\Downloads\Thesis\03 Data\01 Source\czech_mortgages_dataset_v2.csv', 
                 decimal = ',',
                 delimiter = '|',
                 encoding = 'cp437')

# Variable name mapping dataset.
variable_mapping = pd.read_excel(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\mapping_tables.xlsx',
                                 sheet_name = 'variables')

# Interim output library path.
interim_library_path = r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim'

# Programs library path.
programs_library_path = r'C:\Users\JF13832\Downloads\Thesis\02 Programs'

# Set seed.
set_seed = 130816

#------------------------------------------------------------#
# STEP 3: code runner                                        #
#------------------------------------------------------------#

# Prerequisite: run the script above.
def run_scripts(set_seed):

    ''' script 1-1: data wrangling. '''
    execfile(programs_library_path + r'\01-1_-_data_preprocessing.py')

    ''' script 1-2: train-test split '''
    execfile(programs_library_path + r'\01-2_-_train_test_split.py')

    ''' script 1-3: first-round feature exclusions '''
    #execfile(programs_library_path + r'\01-3_-_first-round_feature_exclusions.py')

    ''' script 1-4: oversampling '''
    execfile(programs_library_path + r'\01-4_-_oversampling.py')
    
    ''' script 2-1: binning & woe '''
    execfile(programs_library_path + r'\02-1_-_woe_binning.py')
    
    ''' script 2-2: remove multicollinearity '''
    execfile(programs_library_path + r'\02-2_-_multicollinearity_removal.py')

    ''' script 2-3: logistic regressions '''
    execfile(programs_library_path + r'\02-3_-_logit.py')

run_scripts(set_seed = set_seed)





