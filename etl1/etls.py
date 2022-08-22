import pandas as pd
import numpy as np

class transform_predict:

    def __init__(self):
        self.variables = ['var1','var2','var3']

    def transform_predict(self,x):
        
        __dic__ = {}
        # etls model 1
        y = x.copy()
        for feat in self.variables:
            if feat not in y.columns:
                y.loc[:,feat] = np.nan
        
        __dic__['transformedVariable'] = y[self.variables[:1].div(y[self.variables[-1]].max(axis=1), axis=0)

        return __dic__
