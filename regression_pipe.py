#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 23:56:28 2023

@author: ric
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from datetime import datetime

from kana_tools import famd_prince,preprocessing,reduction_tools,numeric_corr
from kana_tools import CategoricalCorrelation
from ml_kana_tools import reg_features_selection

t_start = datetime.now()

p_test = r'/home/ric/Documents/pycode/py_code/home-data-for-ml-course/test.csv'
p_train = r'/home/ric/Documents/pycode/py_code/home-data-for-ml-course/train.csv'

df_test = pd.read_csv(p_test)
df_train = pd.read_csv(p_train)

class oneHotencodeur:
    def __init__(self, df, columns):
        self.columns = columns
        self.encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in self.columns}

    def transform(self, df):
        df_encoded = pd.DataFrame()
        for col in self.columns:
            df_encoded[col] = self.encoders[col].transform(df[col].astype(str))
        return df_encoded

    def fit(self, df):
        self.correlations = pd.DataFrame(index=self.columns, columns=self.columns, dtype=float)
        df_encoded = self.transform(df)
        return df_encoded
    
class Preprocessing:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=42)

    def separate_variables(self):
        numeric_mask = self.X_train.select_dtypes(include=['int','float']).columns
        self.X_numeric = self.X_train[numeric_mask].fillna(0)
        catego_mask = self.X_train.select_dtypes(include=['object','string']).columns
        self.X_catego = self.X_train[catego_mask]
        self.y_train = self.y_train.to_numpy().ravel()
        self.y_test = self.y_test.to_numpy().ravel()

    def fill_missing_values(self):
        num_manquante_transformer = SimpleImputer(strategy="constant",fill_value=0)
        cat_manquante_transformer = SimpleImputer(strategy="constant",fill_value="inconnue")
        self.X_numeric_val = num_manquante_transformer.fit_transform(self.X_numeric)
        self.X_numeric = pd.DataFrame(self.X_numeric_val,columns=self.X_numeric.columns)
        self.X_catego_val = cat_manquante_transformer.fit_transform(self.X_catego)
        self.X_catego = pd.DataFrame(self.X_catego_val,columns=self.X_catego.columns)

    def explore_numeric_part(self):
        df_num_correl = numeric_corr(self.X_numeric).corr_numeric() 
        numeric_corr(self.X_numeric).plot_heatmap()
        i_preprocess = preprocessing(self.X_numeric,col_label="YrSold")
        X,Y = i_preprocess.if_labels()
        X_scaled = i_preprocess.centree_reduite(X,b_plot=True)
        self.i_reduc = reduction_tools(df_cible=X_scaled,Y=Y)
        self.X_pca = self.i_reduc.acp_biplot(biplot=True,out_folder=r'/home/ric/Documents/pycode/py_code/home-data-for-ml-course')

    def explore_categorical_part(self):
        i_cat_corr = CategoricalCorrelation(self.X_catego,self.X_catego.columns)
        i_cat_corr.fit(self.X_catego,threshold=0.1)
        self.corr = i_cat_corr.get_correlations()
        i_cat_corr.plot_corr()
        i_cat_corr.corr_sup(threshold=0.1)

    def drop_variables(self):
        self.l_col_catego_filtre = [col for col in self.X_catego.columns if self.X_catego[col].nunique()<10]
        self.l_cat_col_high_mod = [col for col in self.X_catego.columns if self.X_catego[col].nunique()>10]
        print(self.l_col_catego_filtre)
        print(self.l_cat_col_high_mod)
        print("nb de col avec une modalité trop élevé:")
        print(len(self.l_cat_col_high_mod))
        print("Nb de variable categorielle retenu pour l'analyse:")
        print(len(self.l_col_catego_filtre))

    def ohe(self,X_catego,X_numeric,X_test,l_col_catego_filtre):
        X_train_cat_new = X_catego[l_col_catego_filtre]
        X_train_cat_new=oneHotencodeur(X_train_cat_new,X_train_cat_new.columns).fit(X_train_cat_new)
        X_test_cat_new = X_test[l_col_catego_filtre]
        X_test_cat_new=oneHotencodeur(X_test_cat_new,X_test_cat_new.columns).fit(X_test_cat_new)
        X_train_cat_new=X_train_cat_new.reset_index(drop=True)
        X_train_new = pd.concat([X_numeric,X_train_cat_new],axis=1,sort=False)
        numeric_mask = X_test.select_dtypes(include=['int','float']).columns
        X_test_num = X_test[numeric_mask]
        X_test_num = X_test_num.fillna(0)
        X_test_cat_new = X_test_cat_new.reset_index(drop=True)
        X_test_num = X_test_num.reset_index(drop=True)
        X_test_new = pd.concat([X_test_num,X_test_cat_new],axis=1,sort=False)
        return X_train_new,X_test_new

print("=============== model selection =====================================")

from sklearn.model_selection import GridSearchCV

class ModelSelection:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def random_forest_selection(self):
        print("=============== model selection =====================================")
        param_grid = [
            {'n_estimators':[100,300,400,800,1000],'random_state':[0,0,0,0,0]},
            {'bootstrap':[False],'n_estimators':[900,1300],'max_features':[30,15]}
        ]
        forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_absolute_error')
        grid_search.fit(self.X_train,self.y_train)
        print(">>>>>>>>>>>>>>>>>>> grid_search.best_params_:")
        print(grid_search.best_params_)
        print('(">>>>>>>>>>>>>>>>>>> les meilleurs hyper parametres:')
        print(grid_search.best_estimator_)
        print(">>>>>>>>>>>>>>>>>>> les RMSE associés:")
        cv_res = grid_search.cv_results_
        for mean_score ,params in zip(cv_res['mean_test_score'],cv_res["params"]):
            print(np.sqrt(-mean_score),params)
        
        best_params = grid_search.best_params_
        model = RandomForestRegressor(**best_params)
        return self.score_model(model)
    
    def xgboost_selection(self):
        param_grid_xgb = {'n_estimators':[100,300,1400,800,1000],
                          'random_state':[0,0,0,0,0],
                          'learning_rate':[0.05,0.07,0.1,0.05,0.05]}
        xgb_reg = XGBRegressor()
        grid_search = GridSearchCV(xgb_reg,param_grid_xgb,cv=5,scoring='neg_mean_absolute_error')
        grid_search.fit(self.X_train,self.y_train)
        print(">>>>>>>>>>>>>>>>>>> grid_search.best_params_:")
        print(grid_search.best_params_)
        print('(">>>>>>>>>>>>>>>>>>> les meilleurs hyper parametres:')
        print(grid_search.best_estimator_)
        print(">>>>>>>>>>>>>>>>>>> les RMSE associés:")
        cv_res = grid_search.cv_results_
        for mean_score ,params in zip(cv_res['mean_test_score'],cv_res["params"]):
            print(np.sqrt(-mean_score),params)

        best_params = grid_search.best_params_
        model = XGBRegressor(**best_params)
        return self.score_model(model)
    
    def score_model(self, model):
        print("================ train with de the best selected model and take a look at his MAE:")
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)
        model.fit(self.X_train, self.y_train.ravel())
        preds = model.predict(self.X_test)

        return mean_absolute_error(self.y_test, preds)
print("============== prediction sur le jeu de test et evaluons la perf du model sur les jeux de test:")

t_end = datetime.now()

print(f"le temps d'execution du script est de {t_end-t_start}")


