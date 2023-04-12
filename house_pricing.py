#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:37:30 2023

@author: ric
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import datetime

from kana_tools import famd_prince,preprocessing,reduction_tools,numeric_corr
from kana_tools import CategoricalCorrelation,oneHotencodeur
from ml_kana_tools import reg_features_selection
from sklearn.preprocessing import StandardScaler



t_start = datetime.now()


p_test = r'/home/ric/Documents/pycode/py_code/home-data-for-ml-course/test.csv'
p_train = r'/home/ric/Documents/pycode/py_code/home-data-for-ml-course/train.csv'

df_test = pd.read_csv(p_test)
df_train = pd.read_csv(p_train)


print("======= infos sur le df_test ===========")
print(df_test.dtypes)
print("****************************************")
print(df_test.columns)
print("****************************************")
print(df_test.shape)

print("======= infos sur le df_train ===========")
print(df_train.dtypes)
print("****************************************")
print(df_train.columns)
print("****************************************")
print(df_train.shape)
print("****************************************")
#cela permet de voir les statistiques des variables numerique et leur nombre:
print(df_train.describe())


print("=============== drop certaines variables (retrospectivement) ============================")
df_train.drop(["GarageCond","ExterQual","GarageYrBlt","GarageArea","1stFlrSF","GrLivArea","YearRemodAdd"],inplace=True,axis=1)
#split data train en test et train
print("======= split df_train =================")
y = df_train[["SalePrice"]]
X = df_train.drop(columns=["SalePrice"])
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=42)
# separation des variables continues et categorielles
numeric_mask = X_train.select_dtypes(include=['int','float']).columns
X_numeric = X_train[numeric_mask]
X_numeric = X_numeric.fillna(0)
print(X_numeric)
catego_mask = X_train.select_dtypes(include=['object','string']).columns
X_catego = X_train[catego_mask]
print(X_catego)
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

print('*****************$$$$$$$')
print(y_test.shape)
print(X_test.shape)
print("======= gestion des valeurs manquantes =============")
num_manquante_transformer = SimpleImputer(strategy="constant",fill_value=0)
cat_manquante_transformer = SimpleImputer(strategy="constant",fill_value="inconnue")
X_numeric_val = num_manquante_transformer.fit_transform(X_numeric)
X_numeric = pd.DataFrame(X_numeric_val,columns=X_numeric.columns)
X_catego_val = cat_manquante_transformer.fit_transform(X_catego)
X_catego = pd.DataFrame(X_catego_val,columns=X_catego.columns)
"""
print("======= exploration de la partie numérique =============")
df_num_correl = numeric_corr(X_numeric).corr_numeric() 
numeric_corr(X_numeric).plot_heatmap()
i_preprocess = preprocessing(X_numeric,col_label="YrSold")
X,Y = i_preprocess.if_labels()
X_scaled = i_preprocess.centree_reduite(X,b_plot=True)
i_reduc = reduction_tools(df_cible=X_scaled,Y=Y)
X_pca = i_reduc.acp_biplot(biplot=True,out_folder=r'/home/ric/Documents/pycode/py_code/home-data-for-ml-course')

print("======= exploration de la partie categorielle =============")
i_cat_corr = CategoricalCorrelation(X_catego,X_catego.columns)
i_cat_corr.fit(X_catego,threshold=0.1)
corr = i_cat_corr.get_correlations()
print(corr)
i_cat_corr.plot_corr()
i_cat_corr.corr_sup(threshold=0.1)
"""
print("======= drop les variables avec un nb de modalité sup à 10 =============")

l_col_catego_filtre = [col for col in X_catego.columns if X_catego[col].nunique()<10]
l_cat_col_high_mod = [col for col in X_catego.columns if X_catego[col].nunique()>10]
print(l_col_catego_filtre)
print(l_cat_col_high_mod)
print("nb de col avec une modalité trop élevé:")
print(len(l_cat_col_high_mod))
print("Nb de variable categorielle retenu pour l'analyse:")
print(len(l_col_catego_filtre))

print("========== encodage des colonnes categorielles retenus ============")
X_train_cat_new = X_catego[l_col_catego_filtre]
X_train_cat_new=oneHotencodeur(X_train_cat_new,X_train_cat_new.columns).fit(X_train_cat_new)
print(X_train_cat_new)
X_test_cat_new = X_test[l_col_catego_filtre]
X_test_cat_new=oneHotencodeur(X_test_cat_new,X_test_cat_new.columns).fit(X_test_cat_new)
print(X_test_cat_new)
print("========== fusion categ+numerics ============")
X_train_cat_new=X_train_cat_new.reset_index(drop=True)
X_train_new = pd.concat([X_numeric,X_train_cat_new],axis=1,sort=False)
print('>> new X_train')
print(X_train_new)
print('>> new X_test')
numeric_mask = X_test.select_dtypes(include=['int','float']).columns
X_test_num = X_test[numeric_mask]
X_test_num = X_test_num.fillna(0)
X_test_cat_new = X_test_cat_new.reset_index(drop=True)
X_test_num = X_test_num.reset_index(drop=True)
X_test_new = pd.concat([X_test_num,X_test_cat_new],axis=1,sort=False)
print(X_test_new)
"""
print("=============== features selections ================================")
# i_FS = reg_features_selection()
# i_FS.rfe_approche(X_train_new,X_test_new,y_train,y_test,step=3)


"""

print("=============== model selection =====================================")
"""
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[100,300,400,800,1000],'random_state':[0,0,0,0,0]},
    {'bootstrap':[False],'n_estimators':[900,1300],'max_features':[30,15]}
    
    ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_absolute_error')
grid_search.fit(X_train_new,y_train)
print(">>>>>>>>>>>>>>>>>>> grid_search.best_params_:")
print(grid_search.best_params_)
print('(">>>>>>>>>>>>>>>>>>> les meilleurs hyper parametres:')
print(grid_search.best_estimator_)
print(">>>>>>>>>>>>>>>>>>> les RMSE associés:")
cv_res = grid_search.cv_results_
for mean_score ,params in zip(cv_res['mean_test_score'],cv_res["params"]):
    print(np.sqrt(-mean_score),params)
    
    
    
param_grid_xgb = {'n_estimators':[100,300,1400,800,1000],
                  'random_state':[0,0,0,0,0],
                  'learning_rate':[0.05,0.07,0.1,0.05,0.05]}
    
    
    
xgb_reg = XGBRegressor()
grid_search = GridSearchCV(xgb_reg,param_grid_xgb,cv=5,scoring='neg_mean_absolute_error')
grid_search.fit(X_train_new,y_train)
print(">>>>>>>>>>>>>>>>>>> grid_search.best_params_:")
print(grid_search.best_params_)
print('(">>>>>>>>>>>>>>>>>>> les meilleurs hyper parametres:')
print(grid_search.best_estimator_)
print(">>>>>>>>>>>>>>>>>>> les RMSE associés:")
cv_res = grid_search.cv_results_
for mean_score ,params in zip(cv_res['mean_test_score'],cv_res["params"]):
    print(np.sqrt(-mean_score),params)

"""


print("================ train with de the best selected model and take a look at his MAE:")
def score_model(model, X_t=X_train_new, X_v=X_test_new, y_t=y_train, y_v=y_test):
    from sklearn.metrics import mean_absolute_error
    print(X_t.shape)
    print(y_t.shape)
    print(X_v.shape)
    print(y_v.shape)
    model.fit(X_t, y_t.ravel())
    preds = model.predict(X_v)
    
    return mean_absolute_error(y_v, preds)

model = XGBRegressor(n_estimators=1400,learning_rate=0.05,random_state=0)
score = score_model(model)
print(" le score(MAE) obtenu par le meilleur model est:")
print(score)

print("============== prediction sur le jeu de test et evaluons la perf du model sur les jeux de test:")

t_end = datetime.now()

print(f"le temps d'execution du script est de {t_end-t_start}")


