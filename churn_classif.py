# -*- coding: utf-8 -*-
'''
Created on 9 avr. 2023

@author: karl
'''


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

from kana_tools import CategoricalCorrelation,numeric_corr
from ml_kana_tools import classif_result
#df_churn = pd.read_csv(r'/home/ric/Telechargements/Churn_Modelling.csv',sep=",") #desktop linux mint
df_churn = pd.read_csv(r'C:\Users\karl\Documents\datasets\Churn_Modelling.csv',sep=",") # desktop win10
print(df_churn.dtypes)
path_out = r'C:\Users\karl\Documents\datasets'
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


def create_model(x_train,y_train,optimizer='adam', init='glorot_uniform'):
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

def param_search(x_train,y_train):
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    from keras.layers import Dense
    from keras.models import Sequential
    xtrain,xvalid,ytrain,yvalid = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    
    classifier = Sequential()
    classifier.add(Dense(units=6,kernel_initializer="glorot_uniform",activation='relu',input_dim=x_train.shape[1]))
    classifier.add(Dense(units=6,kernel_initializer="glorot_uniform",activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer="glorot_uniform",activation='sigmoid'))
    classifier.compile(optimizer="adam",loss = 'binary_crossentropy',metrics=["accuracy"])
    callback_a = ModelCheckpoint(filepath=path_out+'\best_rna_model_churn.hdf5',monitor='binary_crossentropy',save_best_only=True,save_weights_only=True)
    callback_b = EarlyStopping(monitor='binary_crossentropy',mode='min',patience=20,verbose=1)
    historique = classifier.fit(xtrain,ytrain,validation_data=(xvalid,yvalid),epochs=300,batch_size=10,callbacks=[callback_a,callback_b])
    print(">> history['accuracy']")
    print(historique.history['accuracy'])
    print(">> history['val_accuracy'] ")
    print(historique.history['val_accuracy'])
    plt.plot(historique.history['accuracy'])
    plt.plot(historique.history['val_accuracy'])
    plt.title('Exactitude du modele:')
    plt.ylabel('Accuracy')
    plt.xlabel('epochs')
    plt.legend(['training data','validation data'],loc='lower right')
    plt.show()
    plt.plot(historique.history['loss'])
    plt.plot(historique.history['val_loss'])
    plt.title('Perte du modele:')
    plt.ylabel('Perte')
    plt.xlabel('epochs')
    plt.legend(['training data','validation data'],loc='lower right')
    plt.show()

def predict(classifier,x_test,y_test):
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
    



def train_xgb_model(x_train,y_train):
    """
    Training du model xgboost

    Parameters:
    x_train: pandas DataFrame
        Training data features.
    y_train: pandas Series
        Training data labels.

    Returns:
    model: trained XGBoost classification model
    """
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        n_estimators=100,
        objective='binary:logistic',
        subsample=0.8,
        colsample_bytree=0.8,
        seed=42
    )
    model.fit(x_train, y_train)
    return model

def predict_xgboost(classifier,x_test):
    """
    partie prediction.

    """
    predictions = classifier.predict(x_test)
    return predictions



def global_run(df_churn,meth_to_run=""):
    
    X = df_churn.iloc[:,3:13]
    y =  df_churn.iloc[:,13]
    print(X.dtypes)
    l_cat_col = ["Geography","Gender"]
    l_num_col = ["CreditScore","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
    print(X.shape)
    print(round(X[l_num_col].describe(),2))
    df_new = encode_ohe(X, l_num_col, l_cat_col)
    df_new["Exited"] = y
    print(df_new)
    x_train,x_test,y_train,y_test = split_data(df_new,["Exited"])
    print(x_train)
    print(x_train.shape)
    try:
        i_corr_cat = CategoricalCorrelation(df_churn,l_cat_col)
        i_corr_cat.fit(df_churn,threshold=0.3)
        i_corr_cat.corr_sup(threshold=0.3)
    except:
        print("pas de correlation superieur a 0.5 entre les variables selectionnees")
    try:
        i_corr_num = numeric_corr(X[l_num_col])
        i_corr_num.visu_data()
        i_corr_num.corr_numeric()
        i_corr_num.plot_heatmap()
    except:
        print("il semble qu'on a pas suffisament de colonne correles!")
        
    if meth_to_run=="rna":
        # preprocessing des donnees:
        
        #train et predict du model de classification:
        classi =train_rna_model(x_train, y_train)
        predict(classi,x_test)
        
        
    elif meth_to_run=="cv_rna": 
        # Creer un dictionnaire des hyperparametres a tester
        param_grid = {'optimizer': ['rmsprop', 'adam'], 'init': ['glorot_uniform', 'normal', 'uniform'], 'epochs': [50, 100, 150], 'batch_size': [5, 10, 20]}
        
        # Creer un objet KerasClassifier
        model = KerasClassifier(build_fn=create_model)
        
        # Creer un objet RandomizedSearchCV
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10)
        
        # Ajuster le modele aux donnees d'entrainement
        grid.fit(np.array(x_train), np.array(y_train))
        
        # Afficher les meilleurs parametres
        print(grid.best_params_)
    elif meth_to_run=="rna_parms":
        param_search(x_train, y_train)    
    elif meth_to_run=="reg_log":
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(C=0.01,solver='liblinear',random_state=0)
        log_reg.fit(x_train,y_train)
        y_pred_test = log_reg.predict(x_test)
        print("print y_pred_test:")
        print(y_pred_test)
        a_prob = log_reg.predict_proba(x_test)
        print("Model accuracy score pour le model de regression logistique:")
        print(accuracy_score(y_test,y_pred_test))
        print("verification des proportions de sortie ou non:")
        print(y_test.value_counts())
        print("matrice de confusion:")
        classif_result(y_test,y_pred_test).cm()
    
    elif meth_to_run=="xgboost":
        model =train_xgb_model(x_train,y_train)
        y_pred = predict_xgboost(model, x_test) 
        acc = accuracy_score(y_test,y_pred)  
        print("accuracy du model:")
        print(acc)
#===================================== main ======================================================================
global_run(df_churn,meth_to_run="xgboost")   