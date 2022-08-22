import pandas as pd
import numpy as np
import lightgbm
import pickle

class transform_predict:

    def __load_file__(self,nome_arquivo):
        with open(nome_arquivo, 'rb') as input:
            objeto = pickle.load(input)
        return objeto

    def __init__(self,path='./'):

        self.path = path
        self.modelRisk = self.__load_file__(self.path+'model.pkl')
        self.varsRisk = self.__load_file__(self.path+'variables.pkl')

    def transform_predict(self,x):
        
        __dic__ = {}
        # etls model 1
        y = x.copy()
        for feat in self.varsRisk:
            if feat not in y.columns:
                y.loc[:,feat] = np.nan
        y[self.varsRisk] = y[self.varsRisk].div(y[self.varsRisk].max(axis=1), axis=0)
        y = y[self.varsRisk]
        y[self.varsRisk] = y[self.varsRisk].div(y[self.varsRisk].max(axis=1), axis=0)
        __dic__['model1'] = self.modelRisk.predict(y[self.varsRisk])[0]
        __dic__['model1Version'] = '202208_1.0'

        return __dic__
