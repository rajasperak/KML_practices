# KML_practices

![data](https://img.freepik.com/vecteurs-libre/illustration-rpa-design-plat-dessine-main_23-2149277643.jpg?size=626&ext=jpg)

**k** (karl) **ML**(machine learning): 
======================================
- churn_classif.py: preprocessing, approche avec un modele adaboost, gradientBoosting et reseau de neuronne.
- churn_classification_model_exploration.ipynb: exploration des divers modeles et hyperparametres pour optimiser les resultats de la classification.
- churn_model_choice.ipynb: notebook de bunchmark entre les models

**contenus**:
============
dataset qui ont déjà été travaillé et nettoyés
-  dataset house pricing pour la regression
-  dataset detection de  pour la classification.
-  dataset classification des résiliation

**details**:
============

- churn_classif.py:

-------------------
script dans un ide au lieu d'un notebook, en vue d'une industrialisation du traitement et analyses.
cela donne également une bonne pratique pour rendre le code le plus générique possible.

. fonction encode_ohe, va me servir pour remplir par zero les donnnées manquantes. Strategie choisi en fonction de l'objectif de l'analyse.
je préfère également réduire et centrer les données pour qu'elles soient à la meme échelle, pour les modèles basés sur les réseaux de neuronnes
cette étapes est necessaire. Pour les modèles d'ensemble learning utilisé ci-dessous, cette étape ne l'est pas.

. Je suis parti sur une regression logistique pour commencer. Un choix de solver "saga" et "liblinear" a été fait. J'ai fais le choix d'un solver 
"saga" en premier lieu, afin  d'avoir à la fois une régularisation L1 et L2 . Cela permetra à mon modèle d'eviter au maximum le surapprenti-
ssage, vu les dimenssions des données. Après comparaison, des métriques de la classification, le solver liblinear avec l'innverse de alpha à
0.01 est tout aussi satisfesant, voire légèrement meilleur.

. Un modele de gradientBoosting de chez XGBoost a été choisi avec des hyper-paramètres bien choisi après un CV.

. Mais ce script a surtout pour vocation de comparer ces premiers modèles avec le meilleur modèle PML que je pu trouvé. La fonction param_search
sera utilisé pour trouver les meilleurs hyper-paramètres  du réseau avec une seule couche. On a une classification binaire, avec les deux classes
plutôt équilibré en terme de fréquence. la fonction de perte à minimiser par la descente de gradient sera donc 'binary_crossentropy'. 
A l'entrainement du modèle, j'enregistre les meilleurs poids du modèle en minimisant par le Descente de gradient la fonction de cout log_reg 
cité plutôt. Je regarde ensuite le nombre d'epoch minimal où l'optimisation de la perte ne donne plus rien.
Après ceci, j'aurais les hyper-paramètre nécessaire pour le fournir à train_rna_model, valider le modele et regardé sa performance.


- chun_classification_model_exploring.ipynb:

--------------------------------------------

phase exploratoire des divers models et tunning des hyper-parametres:
. modele random forest
. modele adaboost (arbres de decision,regression logistique,...)
. modele gradientBoosting (implémentation xgboost)


- churn_model_choice.ipynb:

---------------------------

. validation croisée sur le pipeline de modèle, composé d'une regression logistique avec differente fonction de pénalité (même si le solver lbfgs n'est 
pas compatible avec la régularisation L1), un classifier random forest avec differente critère de qualité de l'elagage, en l'occurence, je choisi plutot
les critère de gini ou l'entropy. Deux derniers classifier, adaboost et xgboost. plus de détails dans le fichier.
Parmis ces modèles et ces paramètres, le meilleur modèle d'après la validation croisée est le
{'classifier': RandomForestClassifier(min_samples_split=10, n_estimators=500),
 'classifier__criterion': 'gini',
 'classifier__max_depth': None,
 'classifier__max_features': 'sqrt',
 'classifier__min_samples_split': 10,
 'classifier__n_estimators': 500}
 
En appliquant ce dernier sur les données de test, on a les résultats suivant:
Justesse de la Classification (accuracy) : 0.8710
Erreurs de Classification : 0.1290
Recall ou Sensitivity : 0.8857
taux de faux positive:
0.2247191011235955
False Positive Rate : 0.2247
Specificity : 0.7753
 
                precision    recall  f1-score   support

           0       0.89      0.96      0.92      1595
           1       0.78      0.51      0.62       405
           
 Specificity  
 on a une assez bonne precision pour la classe 0, un peu moins pour la classe 1.
 autrement dit la proportion des prédictions correctes de l'appartenance à la classe 0 est plutôt bonne. légèrement moins pour la classe 1.
 Cependant, bien que 78% des prédictions d'appartenance à 1 sont correctes, seulement 51% des classe 1 sont correctement identifiés.
 Ce qui est, au vu du deséquilibre des supports, il n'est pas difficile d'avoir moins de faux positif ou de faux classé à 1, donc plus
 de précision mais moins de recall car on a que 405 individus dans la classe A pour le jeu de test.
 
 >> Axe d'amélioration, échantillonage stratifié dans le split et l'entrainement du modele.
 
