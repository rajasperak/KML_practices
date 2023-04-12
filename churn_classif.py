# -*- coding: utf-8 -*-
'''
Created on 9 avr. 2023

@author: karl
'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

df_churn = pd.read_csv(r'/home/ric/Téléchargements/Churn_Modelling.csv',sep=",")
print(df_churn.dtypes)

def encode_ohe(df,numeric_cols,categorical_cols):
    """
    fonction qui permet d encoder les variables categorielles et de standardiser les variables numerique
    a travers un pipeline.    
    
    Parameters
    ----------
    df : TYPE dataframe
    numeric_cols : liste
        liste de nom de colonne de type numerique
    categorical_cols : liste
        liste de nom de colonne de type categorielle

    Returns
    -------
    ohe_encode_df : dataframe
        dataframe valeurs numeriques standardisees et colonnes categorielles encodees a chaud
        
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='constant',fill_value=0)),
        ('scaler',StandardScaler())
        ])
    
    categ_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='constant',fill_value='inconnue')),
        ('ohe',OneHotEncoder(handle_unknown='ignore'))
        ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num',numeric_transformer,numeric_cols),
            ('cat',categ_transformer,categorical_cols)
        ])
    
    preprocessor.fit(df)  # fit the ColumnTransformer object on the input data
    
    ohe_columns = list(preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_cols))
    new_columns = numeric_cols + ohe_columns
    ohe_encode_df = pd.DataFrame(preprocessor.transform(df), columns=new_columns)
    return ohe_encode_df 

def split_data(df,y_col):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(df.loc[:, ~df.columns.isin(y_col)],df[y_col],test_size=0.2,random_state=0)
    return x_train,x_test,y_train,y_test

def train_rna_model(x_train,y_train,optimizer="adam",init="uniform"):
    """
    

    Parameters
    ----------
    x_train : dataframe
        donnee d entrainement sans les labels.
    y_train : dataframe
        labels d'entrainement.
    optimizer : TYPE string, optional
        DESCRIPTION. la methode d'optimisation utilisee pour la fonction de perte, descente de gradiente'
    init : TYPE string, optional
        DESCRIPTION. initialisation des poids

    Returns
    -------
    classifier : TYPE model
        DESCRIPTION. model de neuronne entraine, on a choisi l'accuracy comme metric de sortie ici'

    """
    from keras.layers import Dense
    from keras.models import Sequential
    print(x_train.values)
    
    classifier = Sequential()
    classifier.add(Dense(units=6,kernel_initializer=init,activation='relu',input_dim=x_train.shape[1]))
    classifier.add(Dense(units=6,kernel_initializer=init,activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer=init,activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss = 'binary_crossentropy',metrics=["accuracy"])
    classifier.fit(x_train.values,y_train.values,batch_size=10,epochs=100)
    return classifier


def create_model(optimizer='adam', init='glorot_uniform'):
    '''

    Returns
    -------
    classifier : TYPE
        DESCRIPTION.meme chose que train_rna_model mais pour le pipeline de gridsearchCV/randomizedSearchCV

    '''
    classifier = Sequential()
    classifier.add(Dense(units=12,kernel_initializer=init,activation='relu',input_shape=(13,)))
    classifier.add(Dense(units=12,kernel_initializer=init,activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer=init,activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss = 'binary_crossentropy',metrics=["accuracy"])
    classifier.fit(x_train,y_train,batch_size=10,epochs=100)
    return classifier


def predict(classifier,x_test):
    """
    

    Parameters
    ----------
    classifier : TYPE model
        DESCRIPTION.
    x_test : TYPE dataframe
        DESCRIPTION. donnee de validation

    Returns
    -------
    print de l'accuracy de la matrice de confusion de la classification.

    """
    from sklearn.metrics import confusion_matrix,accuracy_score
    y_pred = classifier.predict(x_test)
    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    print("la matrice de confusion du model est:")
    print(cm)
    acc = accuracy_score(y_test,y_pred)
    print(f"l'accuracy du model est:{acc}")
    







# preprocessing des donnees:

X = df_churn.iloc[:,3:13]
y =  df_churn.iloc[:,13]
print(X.dtypes)
l_cat_col = ["Geography","Gender"]
l_num_col = ["CreditScore","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
print(X.shape)
df_new = encode_ohe(X, l_num_col, l_cat_col)
df_new["Exited"] = y
print(df_new)
x_train,x_test,y_train,y_test = split_data(df_new,["Exited"])
print(x_train)
print(x_train.shape)
# train et predict du model de classification:
#classi =train_rna_model(x_train, y_train)
#predict(classi,x_test)



# Cr�er un dictionnaire des hyperparam�tres � tester
param_grid = {'optimizer': ['rmsprop', 'adam'], 'init': ['glorot_uniform', 'normal', 'uniform'], 'epochs': [50, 100, 150], 'batch_size': [5, 10, 20]}

# Cr�er un objet KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Cr�er un objet RandomizedSearchCV
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10)

# Ajuster le mod�le aux donn�es d'entra�nement
grid.fit(np.array(x_train), np.array(y_train))

# Afficher les meilleurs param�tres
print(grid.best_params_)
