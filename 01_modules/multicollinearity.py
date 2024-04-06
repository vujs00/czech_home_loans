# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
from collinearity import SelectNonCollinear
from sklearn.feature_selection import f_classif

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


def remove_multicollinearity(df_train, df_test, target):

    # Define selector.
    selector = SelectNonCollinear(correlation_threshold = 0.5, 
                                  scoring = f_classif)
    
    # Prepare X and y.
    y_train = df_train.loc[:, target]
    X_train = df_train.loc[:, df_train.columns != target]
    
    X_train_array = np.nan_to_num(X_train)
    y_train_array = y_train.to_numpy()
    
    selector.fit(X_train_array, y_train_array)
    mask = selector.get_support()
    
    X_train = pd.DataFrame(X_train_array[:, mask], columns = np.array(list(X_train))[mask])
    
    # Repack into data frame.
    # Create a helper in order to retain the y variable in the operations below.
    helper = [[1]]
    helper = pd.DataFrame(helper, columns = [target])
    helper = pd.concat([helper, X_train])
    helper = list(helper)
    
    df_train = df_train[helper]
    df_test = df_test[helper]
            
    return df_train, df_test



