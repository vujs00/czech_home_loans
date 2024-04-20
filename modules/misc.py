# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:27:07 2024

@author: JF13832
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:04:15 2024

@author: JF13832
"""

# Class

from yaml import full_load, safe_dump

class Config:
    
    def __init__(self, path:str) -> None:
        self.path:str = path
        
    def load_cfg_file(self) -> None:
        try:
            with open(self.path) as f:
                cfg = full_load(f)
            return cfg
            print(f"Configuration file loaded from path:{self.path}")
        except:
            raise FileNotFoundError(f'File not found in path:{self.path}')
     

# dataclass

from dataclasses import dataclass 

@dataclass(frozen = True)
class Models:
    y_train: str
    y_test: str
    x_train: str
    x_test: str
    
    @property
    def split_dataframe(self):
        return (self.x_train, self.x_test, self.y_train, self.y_test)
        
    def select_features(self, metric):
        
        sfs = SequentialFeatureSelector(metric,
                                        n_features_to_select = 15,
                                        direction = 'backward',
                                        cv = 5)
        # Fit.
        sfs.fit(self.x_train, self.y_train)

        # Update X matrices with sfs.
        X_train = sfs.transform(self.x_train)
        X_test = df_test[list(sfs.get_feature_names_out())]

    def logistic_regression(self):
        return self.split_dataframe
   
    def run_all(self):
        self.logistic_regression()















