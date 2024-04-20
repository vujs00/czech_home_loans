# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

# Try np.split to reduce # of variables.

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

import pandas as pd
from sklearn.model_selection import train_test_split

#------------------------------------------------------------#
# STEP 2: split                                              #
#------------------------------------------------------------#

def split_train_test(df, set_seed, target):

    X = df.loc[:, df.columns != target]
    y = df.loc[:, target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                        random_state = set_seed)
    
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    df_train = y_train.join(X_train, how = 'left')
    df_test = y_test.join(X_test, how = 'left')

    return df_train, df_test
    


