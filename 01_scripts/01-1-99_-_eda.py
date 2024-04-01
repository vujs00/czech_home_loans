# -*- coding: utf-8 -*-
"""
@author: Stevan Vujcic

"""

#------------------------------------------------------------#
# STEP 1: setup                                              #
#------------------------------------------------------------#

''' In this code, each section might have a part that needs to be manually 
    edited. '''

import numpy as np
import pandas as pd
import datetime
import scorecardpy as sc

df = pd.read_parquet(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\01_-_df.parquet')

#------------------------------------------------------------#
# STEP 2: default rate summary statistic                     #
#------------------------------------------------------------#

''' Prepare a basic summary statistic. '''
helper_1 = df[['obs_yyyymm']].groupby(df['obs_yyyymm']).agg('count')
helper_1.rename(columns = {'obs_yyyymm' : 'obs_count'}, inplace = True)

helper_2 = df[['obs_yyyymm']].loc[df['default_event_flg'] == 1].groupby(df['obs_yyyymm']).agg('count')
helper_2.rename(columns = {'obs_yyyymm' : 'default_events_count'}, inplace = True)

summary = pd.merge(helper_1,
                   helper_2,
                   how = 'outer',
                   left_on = 'obs_yyyymm',
                   right_on = 'obs_yyyymm').reset_index()

summary['default_rate_pct'] = summary['default_events_count']/summary['obs_count']


summary.to_excel(r'C:\Users\JF13832\Downloads\Thesis\03 Data\02 Interim\02_-_default_rate_overview_python.xlsx',
                 index = False)

del helper_1, helper_2, summary


