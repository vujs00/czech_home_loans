# -*- coding: utf-8 -*-
"""
"""

''' setup '''

import pandas as pd
from optbinning import OptimalBinning

from DataGetter import DataGetter
from Preprocessor import Preprocessor

df0 = pd.read_csv(r'C:\Users\JF13832\Downloads\Thesis\02 Data\01 Source\czech_mortgages_dataset_v2.csv', 
                  decimal = ',',
                  delimiter = '|',
                  encoding = 'cp437')

# Variable name mapping dataset.
var_map0 = pd.read_excel(r'C:\Users\JF13832\Downloads\Thesis\02 Data\01 Source\mapping_tables.xlsx',
                         sheet_name = 'variables')

''' definition '''

def bin_univariate(df_in, var_map_in, x_name, dtype):
    df, var_map = DataGetter(df_in, var_map_in).run()
    
    df_train, df_test, df_oot, performance_summary =\
        Preprocessor(130816, 'default_event_flg', df, False, True,
                     201901, 'woe').run()

    x = df_train[x_name].values
    y = df_train['default_event_flg'].values
    
    optb = OptimalBinning(name=x_name, dtype=dtype, solver="cp")
    optb.fit(x, y)
    optb.status
    optb.splits
    binning_table = optb.binning_table
    binning_table.build()
    binning_table.plot(metric="woe", show_bin_labels =True)
    #binning_table.plot(metric="event_rate")


bin_univariate(df0, var_map0, 'retail_behavioral_score', 'numerical')
bin_univariate(df0, var_map0, 'application_score', 'numerical')    
bin_univariate(df0, var_map0, 'age', 'categorical')    
bin_univariate(df0, var_map0, 'dsti_ratio', 'categorical')    
bin_univariate(df0, var_map0, 'ltv_at_loan_origination_ratio', 'categorical')    

