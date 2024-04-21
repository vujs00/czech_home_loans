# -*- coding: utf-8 -*-
"""
"""

#------------------------------------------------------------#
# STEP 1: libraries                                          #
#------------------------------------------------------------#

from dataclasses import dataclass 

import pandas as pd

#------------------------------------------------------------#
# STEP 2: definitions                                        #
#------------------------------------------------------------#


class DataGetter():


    def __init__(self, df, var_map):
        self.df: pd.DataFrame = df
        self.var_map: pd.DataFrame = var_map

    
    def rename_columns(self):
        
        self.df =self.df.rename(columns =\
                                self.var_map.set_index(keys='source_variable_name')['variable_name'])
        return self.df
    
    
    def bool_to_flg(self):
        ''' The function recodes boolean columns to flag. It also updates the
            mapping table to note that certain columns are now flag-types.
            
            Inputs:
            - df: master dataset,
            - var_map: manually created mapping file,
            - interim_library_path: export path for updated version of variable_
                                    mapping dataframe. 
            '''
        
        # Another way of selecting is to take their name from the var_map.
        # This approach is however more robust to typos in that dataframe.
        helper = self.df.select_dtypes(include = 'boolean')
        
        self.df[list(helper)] = self.df[list(helper)].astype(int)
        
        # Recode the variable names to type 'flg'.
        helper = pd.DataFrame(list(helper))
        helper.rename(columns = {0 : 'variable_name'}, inplace = True)
                
        helper['new_variable_name'] = helper['variable_name'].str[:-4] + 'flg' # Automate the recognition of the ending.
        self.df = self.df.rename(columns =\
                                 helper.set_index('variable_name')['new_variable_name'])
    
        self.var_map.loc[self.var_map['analytical_type_cd'] == 'boolean',
                         'variable_name'] =\
            self.var_map['variable_name'].str[:-4] + 'flg'
        
        self.var_map.loc[self.var_map['analytical_type_cd'] == 'boolean',
                         'analytical_type_cd'] = 'flg'
    
        return (self.df, self.var_map)
    
    
    def wrangle_marital_status(self):
        ''' The function re-codes the marital_status variable.
            
            Inputs:
                - df: master dataset,
                - var_map: mapping table. 
                '''

        self.df.loc[self.df['marital_status_cd'].isin(['Svobodn²(ß)']),
                    'marital_status_cd'] = 'single'

        self.df.loc[self.df['marital_status_cd'].isin(['Äenat²',
                                                       'Vdanß',
                                                       'Reg.partner']),
                    'marital_status_cd'] = 'partnered'

        self.df.loc[self.df['marital_status_cd'].isin(['Rozveden²(ß)']),
                    'marital_status_cd'] = 'divorced'

        self.df.loc[self.df['marital_status_cd'].isin(['Vdovec',
                                                       'Vdova']),
                    'marital_status_cd'] = 'widowed'

        self.df.loc[self.df['marital_status_cd'].isin(['Nezadßno',
                                                       'Zem°el(a)']),
                    'marital_status_cd'] = 'other'
        
        self.df.loc[self.df['marital_status_cd'].isnull(),
                    'marital_status_cd'] = 'other'
                
        return self.df
    
    
    def exclude_features(self):
        ''' The function excludes variables according to a manually created
            mapping table.
            
            Inputs:
                - df: master dataset,
                - var_map: mapping table. 
                '''
                
        # The use flag was constructed manually by inspecting the variables.
        helper = self.var_map.loc[self.var_map['use_flg'] ==\
                                  0][['variable_name']]
        self.df = self.df.drop(helper['variable_name'].values.tolist(), axis=1)
        return self.df
    
    
    def run(self):
        
        self.rename_columns()
        self.bool_to_flg()
        self.wrangle_marital_status()
        self.exclude_features()
        
        return (self.df, self.var_map)
    
    