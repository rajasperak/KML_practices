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

- fonction encode_ohe, va me servir pour remplir par zero les donnnées manquantes. Strategie choisi en fonction de l'objectif de l'analyse.
je préfère également réduire et centrer les données pour qu'elles soient à la meme échelle, pour les modèles basés sur les réseaux de neuronnes
cette étapes est necessaire. Pour les modèles d'ensemble learning utilisé ci-dessous, cette étape ne l'est pas.

- Je suis parti sur une regression logistique pour commencer. Un choix de solver "saga" et "liblinear" a été fait. J'ai fais le choix d'un solver 
"saga" en premier lieu, afin  d'avoir à la fois une régularisation L1 et L2 . Cela permetra à mon modèle d'eviter au maximum le surapprenti-
ssage, vu les dimenssions des données. Après comparaison, des métriques de la classification, le solver liblinear avec l'innverse de alpha à
0.01 est tout aussi satisfesant, voire légèrement meilleur.

- Un modele de gradientBoosting de chez XGBoost a été choisi avec des hyper-paramètres bien choisi après un CV.

- Mais ce script a surtout pour vocation de comparer ces premiers modèles avec le meilleur modèle PML que je pu trouvé. La fonction param_search
sera utilisé pour trouver les meilleurs hyper-paramètres  du réseau avec une seule couche. On a une classification binaire, avec les deux classes
plutôt équilibré en terme de fréquence. la fonction de perte à minimiser par la descente de gradient sera donc 'binary_crossentropy'. 
A l'entrainement du modèle, j'enregistre les meilleurs poids du modèle en minimisant par le Descente de gradient la fonction de cout log_reg 
cité plutôt. Je regarde ensuite le nombre d'epoch minimal où l'optimisation de la perte ne donne plus rien.
Après ceci, j'aurais les hyper-paramètre nécessaire pour le fournir à train_rna_model, valider le modele et regardé sa performance.


