# -*- coding: utf-8 -*-
'''
Created on 1 mars 2023

@author: karl
'''

"""
*********************************** descriptions de l'outils  ****************************************************
* - class numeric_corr: destine à analyser la correlation entre les variable numeriques                           *     
*                       dans un dataframe                                                                         *
* - class indep_between_two: destine à analyser si dependance ou no il y a entre deux colonnes d'un               *
*                            dataframe de valeur numeriques.                                                      *
* - class prepocessing: destine à preparer le dataframe et à avoir une première idee de ce qu'il y a              *
*                       à l'interieur. il y a quelques fonction qui permet egalement de faire quelques nettoyage  *
*                       ajustements.                                                                              *
* - class reduction_tools : destine à reduire le nombre de variables à analyser, eliminer les variables correlees,* 
*                           pour eventuellement lancer des modèles dessus, pour de meilleurs resultats.           *
* - class multiv_ana_tools: destine à creer des tables disjonctifs, des tables de burt, des tests de correlation  *
*                           entre des variables mixtes. On aura des outils d'analyse factorielle comme l'ACM,     *
*                           l'AFDM.                                                                               *
* - class clustering: destine à trouver des clusters eventuels. lire la description dans la classe.               *
******************************************************************************************************************
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime
import random,os,itertools
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef

#from factor_analyzer import FactorAnalyzer

class input:
    def __init__(self,path="",index="",separateur_csv=";"):
        self.path_data = path
        if index!="":
            self.data = pd.read_csv(self.path_data,sep=separateur_csv,index_col=index,encoding='latin-1')
        else:
            self.data = pd.read_csv(self.path_data,sep=separateur_csv,encoding='latin-1')
        print(self.data)
        print(self.data.dtypes)
        print(self.data.shape)
    def get_df_cible(self):
        return self.data
        
        
class indep_between_multiple_var(object):
    
    """
        https://asardell.github.io/statistique-python/
        choisir plus de deux variables quantitatives et verifier s'ils ont une relation lineaire signaficative.
        le R2 est egale au coef de correlation r2 de pearson dans le cas d'une regression simple.
        ce test n'est donc utile que dans le cadre d'une regression multiple.
        attention, le R2 depend bcp de l'etendu de la variables a expliquer.
        H0: pente nulle, H1: non nulle et significative, relation entre les 2 variables, avec R2(coef de determination)
        ou
        H0 : Variables independantes si p-value > 5%
        H1 : Variables non independantes si p-value < 5%
        input: dataframe avec les noms des 2 colonnes
        output: R2, + significativite (p-value)
        
    """
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        
    def indep(self):
        from scipy.stats import kendalltau, spearmanr, pearsonr
        pearson_pv = pearsonr(self.X,self.Y)
        spearman_pv = spearmanr(self.X,self.Y)
        kendal_pv = kendalltau(self.X,self.Y)
        return pearson_pv, spearman_pv, kendal_pv
    
class cross_tab(object):
    """
        outils de visualisation
        rajout à faire: ajouter la possibilite de prendre un tableau sans var1 et var2
        et prendre directement les modalites
    """
    def __init__(self,df):
        self.df_ori = df
        
    def t_contingence(self,var1,var2):
        """
            table de contingence avec les modalites de var1 en index et modalites de var2 en colonne
        """
        crosstab_table = pd.crosstab(self.df_ori[var1],self.df_ori[var2])
        print("table de contingence entre les modalite des variable {} et {}".format(var1,var2))
        print(crosstab_table)
        return crosstab_table
    
    def t_contin_sum(self,var1,var2):
        """
            table de contingence avec la somme des modalites de var2 en colonne pour chaque modalite de var1
        """
        sum_crosstab = pd.crosstab(self.df_ori[var1],self.df_ori[var2]).sum()
        print("table de contingence entre les modalite des variable {} et {}, somme sur le dernier:".format(var1,var2))
        print(sum_crosstab)
        return sum_crosstab
    def t_contin_margin(self,var1,var2,agg_margin,margin_name):
        """
            table de contingence avec les modalites de var2 en colonne pour chaque modalite de var1
            cependant les colonnes marginales sont soit la median, la moyenne ou le total
            agg_margin = np.mean,np.median,
        """
        
        cross_tab = pd.crosstab(self.df_ori.var1,self.df_ori.var2,values=self.df_ori.var3,aggfunc=agg_margin,margins_name=margin_name,margins=True)
        print(f"crosstab avec aggregation par la {margin_name}")
        print(cross_tab)
        return cross_tab
    def t_contin_margin(self,var1,var2,agg_margin,margin_name):
        """
            table de contingence avec les modalites de var2 en colonne pour chaque modalite de var1
            cependant les colonnes marginales sont soit la median, la moyenne ou le total
            agg_margin = np.mean,np.median,
        """
        
        cross_tab = pd.crosstab(self.df_ori.var1,self.df_ori.var2,values=self.df_ori.var3,aggfunc=agg_margin,margins_name=margin_name,margins=True)
        print(f"crosstab avec aggregation par la {margin_name}")
        print(cross_tab)
        return cross_tab
    def line_bar_chart(self,var1,var2,graph_type):
        """
            type de graphe: 'pie', 'bar', 'line'
        """
        pd.crosstab(self.df_ori[var1],self.df_ori[var2]).plot(kind=graph_type,figsize=(12,5))

class cross_tab_modality(cross_tab):
    def __init__(self,df):
        super().__init__(df)
        self.df = df
    def t_contin_margin(self):
        print(self.df)
        df = self.df.set_index([self.df.columns[0]])
        print("==============")
        print(self.df.columns[0])
        cross_tab = pd.crosstab(df.index,df.columns,normalize='columns')
        print("crosstab avec aggregation")
        print(cross_tab)
        return cross_tab
        
    
class visualisation(object):
    """
        outils de visualisation
    """ 
class indep_between_two_cat(object):
    """
        choisir deux variables qualitatives et verifier s'ils ont une relation lineaire signaficative.
        faire un test de khi-deux
        input: dataframe avec les noms des 2 colonnes
        H0 : Variables independantes si p-value > 5%
        H1 : Variables non independantes si p-value < 5%
        attention, le test de khi 2 est sensible à la taille de l'echantillon.
        pour completer ce dernier, il est utile de calculer le V de Cramer, pour avoir l'intensite du lien entre les deux variables.
        c de cramer = square(khi2/(taille_echantillon)*(ddl du tableau)) : √(X2/N) / min(C-1, R-1) où C=nb de col et R nb de ligne
        (entre 0.20 et 0.30==> relation moyenne, au dela==>relation forte)
        si dependance il y a, il est aussi interessant de voire a traver un heatmap la source de cette dependance et son intensite.
        on pourra ainsi voir clairement les cases source de dependance (celle qui sont loin de 0)
        https://openclassrooms.com/fr/courses/7410486-nettoyez-et-analysez-votre-jeu-de-donnees/7428635-analysez-deux-variables-qualitatives-avec-le-chi-2
        output: F-stat,p-value,ddl
    """
    def __init__(self,df,A,B):
        self.varA = A
        self.varB = B
        self.df_data = df
    def test_khi2(self,b_t_cont=True):
        from  scipy.stats import chi2_contingency  
        from scipy.stats import chi2
        if b_t_cont:
            contingence = self.df_data[[self.varA,self.varB]].pivot_table(index=self.varA,columns=self.varB,aggfunc=len,margins=True,margins_name="Total")
        else:
            contingence = self.df_data
        print("table de contingence entre {} et {}".format(self.varA,self.varB))
        print(contingence)
        res_khi2 = chi2_contingency(contingence)
        print("===========================================")
        print("sortie test de khi 2:")
        print(res_khi2)
        print("===========================================")
        self.t_cont = contingence
        self.res_khi2 = res_khi2
        
    def v_cramer(self):
        df_wo_tot = self.t_cont.drop(['Total'],axis=1,errors='ignore')
        df_wo_tot = df_wo_tot.drop(['Total'],axis=0,errors='ignore')
        N = np.sum(df_wo_tot.to_numpy())
        ddl = min(df_wo_tot.shape)-1 
        v_cramer = np.sqrt((self.res_khi2[0]/N)/ddl)
        print("============== V-CRAMER  =====================")
        print(f"ddl: {ddl}")
        print(f"N: {N}")
        print(f"khi2:{self.res_khi2[0]}")
        print(f"statistique de cramer: {v_cramer}")
        print("==============================================")
        
    def heat_map_dependance(self):
        import seaborn as sns
        tx = self.t_cont.loc[:,["Total"]]
        ty = self.t_cont.loc[["Total"],:]
        n = len(self.df_data)
        print(f"nombre de ligne du dataframe: {n}")
        indep = tx.dot(ty)/n
        c = self.t_cont.fillna(0)
        mesure = ((c-indep)**2)/indep
        xi_n = mesure.sum().sum()
        table = mesure/xi_n
        sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])
        plt.show()
        
         

class Anova1(object):
    """
        Anova à 1 facteur: pour tester s'il y a une difference significative entre les moyennes des differentes 
        modalites de la variable qualitatives.
        H0: M1=M2...=Mk
        H1: il y a au moins un groupe dont la moyenne theorique (population) est differente des autres.
        pour savoir quel est le groupe qui differe des autres, soit faire un test t-test (Mann-Wihtney) paire à paire.
        pour la methode parametrique, les hypotheses à verifier sont:
        - independance entre les residis
        - normalite des residus
        - homogeneite de la variance
        on mettra que 3 modalites sur ce cas. On surchargera la classe si on aura besoin de plus.
        
        pour le test non-parametrique: au lieu de la moyenne, on aura affaire aux medianes.
    """
    def __init__(self,M1,M2,M3,meth):
        self.meth = meth
        self.M1 = M1
        self.M2= M2
        self.M3 = M3 
    def Anova(self):
        import scipy.stats as stats
        if self.meth == "param":
            res = stats.f_oneway(self.M1,self.M2,self.M3)
            print("resultats de l'ANOVA sous conditions des hypotheses d'application du test :")
            print(res)
        elif self.meth == "nonparam":
            res = stats.kruskal(self.M1,self.M2,self.M3)
            print("resultats de L'anova non parametrique (kruskal-wallis):")
            print(res)
            
            
    
        
class preprocessing(object):
    
    def __init__(self,input=pd.DataFrame(),col_label=""):
        
        self.df_data = input
        self.col_label = col_label
    def if_labels(self):
        self.X = self.df_data[self.df_data.columns.difference([self.col_label])]
        self.Y = self.df_data[self.col_label]
        return self.X,self.Y
    def ana_dist_labels(self,b_plot=False):
        self.df_data.groupby([self.col_label]).size().plot(kind="bar")
        if b_plot:
            plt.show()
    def ana_dist_numerics(self,b_plot=False):
        sort_by_mean = self.X.mean().sort_values(ascending=False)
        print("========= moyenne de chaque variable ========")
        print(sort_by_mean[:5])
        self.X[sort_by_mean.index].plot(kind='box',figsize=(15,4),rot=90,ylabel='distribution des variables')
        if b_plot:
            plt.show()
    def centree_reduite(self,X,b_plot=False):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled,index=X.index,columns=X.columns)
        X_scaled.plot(kind="box",figsize=(15,4),rot=90,ylabel='distribution_normalized')
        if b_plot:
            plt.show()
        return X_scaled
    def get_rid_outliers(self,X,b_plot=False):
        df = X.copy()
        def box_only(s_col):
            q1 = s_col.quantile(0.25)
            q3 = s_col.quantile(0.75)
            iiq = q3-q1
            barr_sup = q3+1.5*iiq
            barr_inf = q1-1.5*iiq
            s_out = s_col.loc[(s_col>barr_inf)&(s_col<barr_sup)]
            return s_out
        df = df.apply(lambda x:box_only(x),axis=1)
        print(df)
        df.plot(kind="box",figsize=(15,4),rot=90,ylabel='distribution wo outliers',showfliers=False)
        if b_plot:
            plt.show()
        return df
    
class reduction_tools(object):
    def __init__(self,meth_choice='pca',df_cible=pd.DataFrame(),Y=pd.DataFrame()):
        self.meth = meth_choice
        self.df_cible = df_cible
        self.labels_series = Y
    def acp(self,b_plot = False):
        
        pca = PCA()
        a_X_pca = pca.fit_transform(self.df_cible)
        print("=============== affichage de a_X_pca ========================")
        print(a_X_pca)
        #convertissons le resultat de l'acp en dataframme et renommer les noms de colonnes de PC1 a PCX
        pca_columns = ["PC"+str(c) for c in range(1,a_X_pca.shape[1]+1,1)]
        X_pca = pd.DataFrame(a_X_pca,index=self.df_cible.index,columns=pca_columns)
        print(X_pca.head(5))
        # variance explique par les composantes
        self.a_explained_variance = pca.explained_variance_ratio_
        explained_variance = pd.Series(dict(zip(X_pca.columns,100*self.a_explained_variance)))
        explained_variance.plot(kind="bar",figsize=(15,4),rot=90,ylabel='variance expliquee (%)')
        if b_plot:
            plt.show()
        return a_X_pca
    def plot_acp_result(self,nb_comp=4,b_3d_plot=True):
        "a rajouter le cercle de correlation"
        if b_3d_plot and nb_comp==3:
            total_var = self.a_explained_variance[:3].sum()*100
            var_pc1 = round(self.a_explained_variance[0]*100,0)
            var_pc2 = round(self.a_explained_variance[1]*100,0)
            var_pc3 = round(self.a_explained_variance[2]*100,0)
            print("total_var:")
            print(total_var)
            if  self.labels_series.empty: 
                fig = px.scatter_3d(
                    self.acp(),
                    x=0,y=1,z=2,
                    title=f'Variance total expliquee par la projection lineaire (ACP): {total_var: .2f}% [PC1({var_pc1}%),PC2({var_pc2}%),PC3({var_pc3}%)]',
                    labels={'0':'PC 1','1':'PC 2','2':'PC 3'}
                
                    )
            else:
                fig = px.scatter_3d(
                    self.acp(),
                    x=0,y=1,z=2,
                    title=f'Variance total expliquee par la projection lineaire (ACP): {total_var: .2f}% [PC1({var_pc1}%),PC2({var_pc2}%),PC3({var_pc3}%)]',
                    labels={'0':'PC 1','1':'PC 2','2':'PC 3'},
                    color = self.labels_series
                
                    )
            plot(fig, auto_open=True,filename="pca_plot_3d.html")
        
        
        else:
            print("pour rappel, pour avoir les 3 PC dans un graphe, nb_comp doit etre à 3!")
            labels = {
                str(i): "PC{} ({})%".format(i+1,round(var,0)) for i, var in enumerate(self.a_explained_variance*100)
                }
            from pprint import pprint
            pprint(labels)
            if not self.labels_series.empty:
                fig = px.scatter_matrix(self.acp(),labels=labels,dimensions=range(nb_comp),color = self.labels_series, title="Projection lineaire (ACP) sur les differentes composantes principales:")
            else:
                fig = px.scatter_matrix(self.acp(),labels=labels,dimensions=range(nb_comp),title="Projection lineaire (ACP) sur les differentes composantes principales:")
            fig.update_traces(diagonal_visible=False)
            plot(fig, auto_open=True,filename="pca_plot_2d.html")
    def Draw_arrow(self,fig,l_X,l_Y,i,label):
        tex_posi = ["top center","top right","middle center","bottom left","middle right"]
        fig.add_trace(go.Scatter(
        x=[l_X[i],0],
        y=[l_Y[i],0],
        mode="lines+markers+text",
        name=str(label[i]),
        text=[str(label[i])],
        textposition=random.choice(tex_posi)
        ))
    def plot_circle(self,l_X,l_Y,label,Xlabel,Ylabel,biplot=True):
        '''
        DESCRIPTION : trace le cercle  avce le choix d avoir le biplot ou pas
        INPUT : -liste des coordonnees de chaque variable
                -label pourcentage inertie
        OUTPUT : retourne fig pour etre utiliser dans la fonction circleofcorrelation
        '''
        trace0 = go.Scatter(
        x= l_X,
        y=l_Y,
        mode = 'markers',
        showlegend=False
        
        )
        
        data = [trace0]
        layout = {
        'xaxis': {
            'range': [-3, 3],
            'zeroline': True,
            'title':Xlabel,    
        },
        'yaxis': {
            'range': [-3, 3],'title':Ylabel,
        },
        'width': 800,
        'height': 800,
        
        'shapes': [
            {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'x0': -1,
                'y0': -1,
                'x1': 1,
                'y1': 1,
                'line': {
                    'color': 'rgba(250, 171, 96, 1)',
                },
            }]}
        
        layout['shapes'][0]['x0']=-1
        layout['shapes'][0]['y0']=-1
        layout['shapes'][0]['x1']=1
        layout['shapes'][0]['y1']=1
        fig=go.Figure(data=data, layout=layout)
        fig.layout.update(showlegend=False)
        
        for i in range(len(l_Y)):
            self.Draw_arrow(fig, l_X, l_Y, i,label)
        if biplot==False:
            plot(fig,filename='graphe des variables')
        else:
            return fig

    def circleOfCorrelations(self,var_contr, cb_var,axes=[0,1],biplot=False):
        '''
            DESCRIPTION : projete les contribution de chaque variable et de chaque individus dans la variabilite
                          du nuage de point.
            INPUT:  -var_contr: contribution de chaque variable, donc coordonnees des 
                    -cb_var: courbe d'inertie
                    
            OUTPUT: -fig pour etre manger par myPCA
        
        '''
        x=[]
        y=[]
        l=[]
    
        for idx in range(len(var_contr[axes[0]])):
            x.append(var_contr[axes[0]][idx])
            y.append(var_contr[axes[1]][idx])
            cible = var_contr[axes[0]][idx]
            index = var_contr[var_contr[axes[0]]==cible].index.values
            l.append(index)
        
        PC0_label= '{:.2f}%'.format(cb_var[axes[0]]*100)
        PC1_label =  '{:.{prec}f}%'.format(cb_var[axes[1]]*100,prec=2)
        fig = self.plot_circle(x,y,l,PC0_label,PC1_label,biplot)
        return fig     
    def hover_text(self,df,nb_hovertext=4):
        
        l_leg=[]
        for index,col in df.iterrows():
            col_sort = col.sort_values(ascending=False)
            col_sort = col_sort.nlargest(nb_hovertext)
        
            formattage = ''
            for key in col_sort.keys():
                br = (str(key)+str(":")+str(col_sort[key]))
                formattage=formattage+'<br>'+br
                l_leg.append(formattage)   
        return l_leg   
    def acp_biplot(self,axes=[0,1],biplot=False,l_label=[],out_folder='',bplot_name='',plot_var=False):
        '''
        descriptif: à utiliser dans le cadre d'une analyse visuelle
        
        INPUT:   -df: dataframe d entree sans colonne index
                -axes: choix des axes, rotation
                -biplot: graphe biplot ou graphe des individu et variable
                -l_label : ce qui est afficher avec les points en les survolants
                            a trouver dans le tableau. voir exemple dans le main.
        OUTPUT:
               -Graphe des individus-variable ou  biplot
        '''
        
        pca = PCA(n_components="mle")
        pca_res = pca.fit_transform(self.df_cible)
        cb_var = pd.Series(pca.explained_variance_ratio_)
        plt.clf()
        cb_var.plot(kind='bar',title=" valeurs propres par ordre decroissante (VP associees aux Vect ppes et aux axes maximisant la variance des donnees):")
        plt.show()
        coef = np.transpose(pca.components_)
        cols = [x for x in range(len(cb_var))]
        var_contr = pd.DataFrame(coef, columns=cols, index=self.df_cible.columns)
        dat = pd.DataFrame(pca_res,columns=cols)
        PC0_label= '{:.2f}%'.format(cb_var[axes[0]]*100)
        PC1_label =  '{:.2f}%'.format(cb_var[axes[1]]*100)
        if biplot:
            fig = self.circleOfCorrelations(var_contr, cb_var,axes,biplot)
            fig.add_trace(go.Scatter(
            x=(dat[axes[0]]/(max(dat[axes[0]].tolist())/2)).tolist(),
            y=(dat[axes[1]]/(max(dat[axes[1]].tolist())/2)).tolist(),
            mode="markers",
            opacity=0.7,
            showlegend=False,
            text=l_label,
            hoverinfo='text',
            
    
            marker=dict(size=9,color='#AF97C2')
    
            ))
            fig.layout.update({
                'xaxis': {
                    'range': [-3,3 ],
                    'zeroline': True,
                    'title':'axe '+str(axes[0])+' avec: '+str(PC0_label),    
            },
                'yaxis': {
                'range': [-3,3 ],'title':'axe '+str(axes[1])+' avec: '+str(PC1_label),
            },
            'width': 800,
            'height': 800,
            'hovermode':'closest'
                })
            fig.layout.update(showlegend=False,title = "ANALYSE EN COMPOSANTE PRINCIPALES (COURBE BIPLOT) :",)
            fig.update_yaxes(automargin=True)
            try: 
                os.makedirs(os.path.join('D:\\',out_folder))
            except OSError:
                if not os.path.isdir(out_folder):
                    print("ce dossier existe deja!!!!")   
            output_path = os.path.join('D:\\',os.path.join(out_folder,bplot_name+"biplot.html"))     
            plot(fig,filename=output_path)
        else:
            fig2=go.Figure()
            fig2.add_trace(        
            go.Scatter(
            x= dat[axes[0]].tolist(),
            y=dat[axes[1]].tolist(),
            mode = 'markers',
            showlegend=False,
            text=l_label,
            hoverinfo='text',
            marker=dict(size=9,color='black')
            ))
            
            fig2.layout.update({
                'xaxis': {
                    'range': [-15,15 ],
                    'zeroline': True,
                    'title':'axe '+str(axes[0]),    
            },
            'yaxis': {
                'range': [-20,20 ],'title':'axe '+str(axes[1]),
            },'hovermode':'closest',
            })
            
            fig2.layout.update(showlegend=False)
            try: 
                os.makedirs(os.path.join('D:\\',out_folder))
            except OSError:
                if not os.path.isdir(out_folder):
                    print("ce dossier existe deja!!!!")   
            output_path = os.path.join('D:\\',os.path.join(out_folder,bplot_name+".html"))     
            plot(fig2,filename=output_path)
            
            
        return pca_res
        
        
        
            
    
    def t_sne(self,nb_comp=2,nb_voisin=45):
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=nb_comp,init='pca',random_state=0,n_jobs = -1,perplexity=nb_voisin)
        a_X_tsne = tsne.fit_transform(self.df_cible)
        #resultat et coordonnees en dataframe
        columns = ["DIM"+str(c) for c in range(1,a_X_tsne.shape[1]+1,1)]
        X_tsne = pd.DataFrame(a_X_tsne,index=self.df_cible.index,columns=columns)
        div_tsne = tsne.kl_divergence_
        return a_X_tsne,X_tsne,div_tsne
    def plot_tsne_result(self,nb_comp=2,nb_voisin=45):
        a_tsne,df_tsne,div_tsne = self.t_sne(nb_comp=nb_comp,nb_voisin=nb_voisin)
        print("******************** tsne ******************")
        print(df_tsne)
        if nb_comp==2 and self.labels_series.empty==False:
            fig = px.scatter(a_tsne,x=0,y = 1,labels={'0':'DIM 1','1':'DIM 2'},color = self.labels_series,title=f"Reduction de dimension probabiliste : avec une divergence de kullback-Leibler de : {div_tsne:.3f}")
            plot(fig, auto_open=True,filename="tSNE_plot_2d.html")
        elif nb_comp==2 and self.labels_series.empty:
            fig = px.scatter(a_tsne,x=0,y = 1,labels={'0':'DIM 1','1':'DIM 2'},title=f"Reduction de dimension probabiliste: avec une divergence de kullback-Leibler de : {div_tsne:.3f}")
            plot(fig, auto_open=True,filename="tSNE_plot_2d.html")
        elif nb_comp==3 and self.labels_series.empty:
            fig = px.scatter_3d(
                    a_tsne,
                    x=0,y=1,z=2,
                    title=f'Reduction de dimension probabiliste (T-sNE): avec une divergence de kullback-Leibler de : {div_tsne:.3f}',
                    labels={'0':'DIM 1','1':'DIM 2','2':'DIM 3'},
                
                    )
            plot(fig, auto_open=True,filename="tSNE_plot_3d.html")
        elif nb_comp==3 and self.labels_series.empty==False:
            fig = px.scatter_3d(
                    a_tsne,
                    x=0,y=1,z=2,
                    title=f'Reduction de dimension probabiliste (T-sNE): avec une divergence de kullback-Leibler de : {div_tsne:.3f}',
                    labels={'0':'DIM 1','1':'DIM 2','2':'DIM 3'},
                    color = self.labels_series
                
                    )
            plot(fig, auto_open=True,filename="tSNE_plot_3d.html")
    def umap_meth(self,nb_comp=2):
        import umap
        embedding = umap.UMAP(n_components=nb_comp,random_state=0,n_jobs=-1)
        a_X_umap = embedding.fit_transform(self.df_cible)
        columns = ['DIM'+str(c) for c in range(1,a_X_umap.shape[1]+1,1)]
        X_umap = pd.DataFrame(a_X_umap,index=self.df_cible.index,columns=columns)
        return a_X_umap,X_umap
    
    
    def plot_umap_result(self,nb_comp=2):
        a_X_umap,X_umap = self.umap_meth(nb_comp=nb_comp)
        print("******************** X_umap ******************")
        print(X_umap)
        if nb_comp==2 and self.labels_series.empty==False:
            fig = px.scatter(a_X_umap,x=0,y = 1,labels={'0':'DIM 1','1':'DIM 2'},color = self.labels_series,title="Reduction de dimension probabiliste (UMAP) :")
            plot(fig, auto_open=True,filename="UMAP_plot_2d.html")
        elif nb_comp==2 and self.labels_series.empty:
            fig = px.scatter(a_X_umap,x=0,y = 1,labels={'0':'DIM 1','1':'DIM 2'},title="Reduction de dimension probabiliste (UMAP) :")
            plot(fig, auto_open=True,filename="UMAP_plot_2d.html")
        elif nb_comp==3 and self.labels_series.empty:
            fig = px.scatter_3d(
                    a_X_umap,
                    x=0,y=1,z=2,
                    title="Reduction de dimension probabiliste (UMAP) :",
                    labels={'0':'DIM 1','1':'DIM 2','2':'DIM 3'},
                
                    )
            plot(fig, auto_open=True,filename="UMAP_plot_3d.html")
        elif nb_comp==3 and self.labels_series.empty==False:
            fig = px.scatter_3d(
                    a_X_umap,
                    x=0,y=1,z=2,
                    title="Reduction de dimension probabiliste (UMAP) :",
                    labels={'0':'DIM 1','1':'DIM 2','2':'DIM 3'},
                    color = self.labels_series
                
                    )
            plot(fig, auto_open=True,filename="UMAP_plot_3d.html")
            
            
    def pacmap_tool(self):
        """ a developper """
    
class numeric_corr(object):
    def __init__(self,df_data):
       
        self.df_num_only = df_data.select_dtypes(include=np.number)
    def visu_data(self):
        print(self.df_num_only)
        print(self.df_num_only.dtypes)
    def corr_numeric(self):
        df_num_only = self.df_num_only.select_dtypes(include=np.number)
        #print(df_num_only)
        df_corr = df_num_only.corr(method='kendall').abs()
        print(df_corr)
        df_corr_unstack = df_corr.unstack()
        df_corr_top = df_corr_unstack.sort_values(kind="quicksort",ascending=False).reset_index()
        df_corr_top.columns = ["col1","col2","coef"]
        df_corr_top = df_corr_top.loc[df_corr_top["col1"]!=df_corr_top["col2"]]
        print(df_corr_top)
        df_corr_top.coef = df_corr_top.coef.round(3)
        print(df_corr_top.head(20))
    def plot_heatmap(self):
        df_corr = self.df_num_only.corr(method='kendall')
        df_corr_top = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
        df_corr_top = df_corr_top.stack().reset_index()
        df_corr_top.columns = ['col1', 'col2', 'coef']
        df_corr_top = df_corr_top[df_corr_top['coef'] > 0.5].replace(1, np.nan)
        df_corr_bottom = df_corr.where(np.tril(np.ones(df_corr.shape), k=-1).astype(bool))
        df_corr_bottom = df_corr_bottom.stack().reset_index()
        df_corr_bottom.columns = ['col1', 'col2', 'coef']
        df_corr_bottom = df_corr_bottom[(df_corr_bottom['coef'] < 0.5)&(df_corr_bottom['coef'] > 0.3)]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(df_corr_top.pivot('col1', 'col2', 'coef'), cmap='coolwarm', annot=True, fmt='.2f', ax=axes[0])
        axes[0].set_title('Variables with correlation > 0.5')
        sns.heatmap(df_corr_bottom.pivot('col1', 'col2', 'coef'), cmap='coolwarm', annot=True, fmt='.2f', ax=axes[1])
        axes[1].set_title('Variables with correlation < 0.5')
        plt.show()

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
    


class CategoricalCorrelation:
    def __init__(self, df, columns):
        self.columns = columns
        self.encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in self.columns}

    def transform(self, df):
        df_encoded = pd.DataFrame()
        for col in self.columns:
            df_encoded[col] = self.encoders[col].transform(df[col].astype(str))
        return df_encoded

    def fit(self, df, threshold=0):
        self.correlations = pd.DataFrame(index=self.columns, columns=self.columns, dtype=float)
        df_encoded = self.transform(df)
        for col1 in self.columns:
            for col2 in self.columns:
                if col1 == col2:
                    self.correlations.loc[col1, col2] = 1.0
                else:
                    coef = matthews_corrcoef(df_encoded[col1], df_encoded[col2])
                    if coef > threshold:
                        self.correlations.loc[col1, col2] = coef
                    else:
                        self.correlations.loc[col1, col2] = 0.0
        self.high_corr_pairs = [(self.correlations.index[i], self.correlations.columns[j]) 
                                for i in range(len(self.columns)) for j in range(i+1,len(self.columns)) 
                                if self.correlations.iloc[i,j] > threshold]
        print("Pairs with correlation coefficient > threshold:")
        print(self.high_corr_pairs)

    def get_correlations(self):
        print("======= get_correlations ========:")
        return self.correlations
    
    def corr_sup(self, threshold):
        # Selectionner les paires avec un coefficient de correlation superieur au seuil
        pairs_sup_seuil = self.correlations[self.correlations > threshold].stack().reset_index()
        pairs_sup_seuil.columns = ['col1', 'col2', 'corr_coef']
        pairs_sup_seuil = pairs_sup_seuil[pairs_sup_seuil['corr_coef'] != 1.0]
    
        # Afficher le nouveau dataframe avec les paires superieures au seuil
        print("Pairs with correlation coefficient > threshold:")
        print(pairs_sup_seuil)
         # Creer un heatmap pour visualiser les correlations entre les paires selectionnees
        fig, ax = plt.subplots(figsize=(len(pairs_sup_seuil['col1'].unique()), len(pairs_sup_seuil['col2'].unique())))
        sns.heatmap(pairs_sup_seuil.pivot('col1', 'col2', 'corr_coef'), vmin=-1, vmax=1, cmap='coolwarm', annot=True, ax=ax)
        ax.set_title('Categorical Correlation Heatmap')
        plt.show()
        return pairs_sup_seuil

    def plot_corr(self):
        fig, ax = plt.subplots(figsize=(len(self.columns), len(self.columns)))
        sns.heatmap(self.correlations, vmin=-1, vmax=1, cmap='coolwarm', annot=True, ax=ax)
        ax.set_title('Categorical Correlation Heatmap')
        plt.show()







class ana_factoriel(object):
    def __init__(self,df_cible,l_col_desact=[],l_row_desact=[],col_index=""):
        self.df = df_cible
        self.l_col_desactive = l_col_desact
        self.l_lig_desactive = l_row_desact
        self.col_index = col_index
        
    def plot_coord(self,df,x='row_coord_dim1',y='row_coord_dim2',color="blue"):
        

        fig = px.scatter(df, x=x, y=y, text=df.index)
        
        fig.update_traces(textposition='top center')
        
        fig.update_layout(
            title={
                'text': "Graphe de l'AFC avec les individus supplementaires:",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title="dim1",
            yaxis_title="dim2",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color=color,
                
            )
        )
        
        
        return fig

    def afc(self):
        from fanalysis.ca import CA
        from scipy.stats import chi2_contingency
        import plotly.graph_objs as go
        import seaborn as sns
        
        if  self.l_col_desactive != []:
            df_actives = self.df[self.df.columns.difference(self.l_col_desactive)]
            df_supplem = self.df.loc[self.l_col_desactive]
        elif self.l_lig_desactive != []:
            
            df_index_to_exclude = self.df.index.isin(self.l_lig_desactive)
            df_actives = self.df[~df_index_to_exclude]
            df_supplem = self.df.loc[self.df.index[self.l_lig_desactive]]
            df_supplem = df_supplem.set_index([self.df.columns[0]])
        else:
            df_actives = self.df
        df_actives.set_index(self.col_index,inplace=True)
        
        print("______ dataframe des variables ou individus actifs:________")
        print(df_actives)
        # instantiation de CA et entrainement
        print("--------valeurs propres:----------")
        i_ca = CA(row_labels=df_actives.index,col_labels=df_actives.columns,stats=True)
        i_ca.fit(df_actives.values)
        #affiche les valeurs propres:
        print(i_ca.eig_)
        #graphique des valeurs propres:
        print("------- plot des valeurs propres ------")
        i_ca.plot_eigenvalues()
        #Analyse des modalites lignes
        print("------ Analyse des modalites lignes-------")
        print(">> recuperation des infos lignes:")
        infos_lignes = i_ca.row_topandas()
        print(infos_lignes.head(5))
        #carte des individus:
        print("cartes des individus:")
        i_ca.mapping_row(num_x_axis=1,num_y_axis=2)
        print("la contribution au premier facteur:")
        i_ca.plot_row_contrib(num_axis=1)
        print("la contribution au deuxieme facteur:")
        i_ca.plot_row_contrib(num_axis=2)
        print(">> rappel des profils lignes en proba:")
        #diviser pour chaque ligne de chaque colonne par la somme de ttes les lignes de la colonne
        profil_lignes = df_actives.divide(df_actives.sum(axis=1),axis=0)
        print(profil_lignes)
        print("profil moyen, barycentre global (origine) des profils lignes, profil marginal")
        sum_tot = np.sum(df_actives.values)
        sum_ligne = df_actives.sum(axis=0)
        profil_marginal = sum_ligne/sum_tot
        print(profil_marginal)
        print("------ Analyse des modalites colonnes-------")
        infos_col = i_ca.col_topandas()
        print(infos_col)
        #carte des colonnes:
        print("cartes des colonnes:")
        i_ca.mapping_col(num_x_axis=1,num_y_axis=2)
        #carte simultanees:
        print("------ Analyse simultanee-------")
        i_ca.mapping(num_x_axis=1,num_y_axis=2)
        print("------ Decomposition du khi2-------")
        print(">> chi2 et table sous independance:")
        res_chi2 = chi2_contingency(df_actives,correction=False)
        print(res_chi2)
        print(res_chi2[3])
        print(">> calcul de l'ecrat a l independance:")
        contribkhi2 = ((df_actives.values - res_chi2[3])**2)/res_chi2[3]
        print(">> la somme des contrib = notre khi2? (verifions):")
        print(np.sum(contribkhi2))
        print(">> dataframe de l'ecart à l'independance:")
        frac_contrib = contribkhi2/res_chi2[0]
        df_contrib = pd.DataFrame(frac_contrib,index=df_actives.index,columns=df_actives.columns)
        print(">> df et heatmap d'ecart à l'independance:")
        print(df_contrib)
        sns.heatmap(df_contrib,vmin=0,vmax=np.max(frac_contrib),cmap="Reds")
        print(">> residus des ecarts à l'independance: pour savoir si repulsion ou attraction")
        residus_std = (df_actives.values-res_chi2[3])/np.sqrt(res_chi2[3])
        df_resi_std  = pd.DataFrame(residus_std,index=df_actives.index,columns=df_actives.columns)
        print(">> df_resi_std:")
        print(residus_std)
        print(df_resi_std)
        sns.heatmap(df_resi_std,center=0.0,cmap=sns.diverging_palette(10,250,as_cmap=True))
        if not df_actives.equals(self.df):
            print("------ rajout des lignes ou colonnes supplementaires -------")
            print(">> calcul des  coordonnees factorielles supplementaire:")
            print(df_supplem.values)
            coord_supp = i_ca.transform(list(df_supplem.values))
            print(coord_supp)
            df_coord_sup = pd.DataFrame(coord_supp,index=df_supplem.index)
            df_coord_sup = df_coord_sup.iloc[:,0:2]
            df_coord_sup = df_coord_sup.rename(columns={0:"row_coord_dim1",1:"row_coord_dim2"})
            df_coord = infos_lignes.iloc[:,0:2]
            df_coord_tot = pd.concat([df_coord,df_coord_sup],axis=0)
            print(df_coord)
            print(df_coord_tot)
            fig = self.plot_coord(df_coord_tot,color="black")
            fig2 = self.plot_coord(infos_col.iloc[:,0:2],x='col_coord_dim1',y='col_coord_dim2',color='green')
            #fig.update_layout(title="Graphes simultanees avec les ind supplementaires:")
            
            fig.add_traces(fig2.data)
            fig.update_layout(title="Graphes simultanees avec les ind supplementaires:")
            plot(fig, auto_open=True,filename="ANALYSE DES CORRESPONDANCES(SORTIE).html")
            
    def mca(self):
        from fanalysis.mca import MCA
        print("ici les data depuis MCA")
        
        df_actives=self.df.set_index(self.col_index)
        
        print(df_actives.values)
        my_mca = MCA(row_labels=df_actives.index.values,var_labels=df_actives.columns.values)
        my_mca.fit(df_actives.values)
        my_mca.plot_eigenvalues()
        my_mca.plot_eigenvalues(type="cumulative")
        print(">>>>>>> les coordonnees des points lignes:(my_mca.row_coord_)")
        print(my_mca.row_coord_)
        print(">>>>>>> les contributions des profils lignes: (my_mca.row_contrib_)")
        print(my_mca.row_contrib_)
        print(">>>>>>> le cos 2 des profils lignes: (my_mca.row_cos2_)")
        print(my_mca.row_cos2_)
        print(">>>>>>> les stats des profils lignes:")
        print(my_mca.row_topandas())
        print(">>>>>>> les coordonnees des variables :(my_mca.col_coord_)")
        print(my_mca.col_coord_)
        print(">>>>>>> les contributions des variables: (my_mca.col_contrib_)")
        print(my_mca.col_contrib_)
        print(">>>>>>> le cos 2 des colonnes: (my_mca.col_cos2_)")
        print(my_mca.col_cos2_)
        print(">>>>>>> les stats des profils lignes:")
        print(my_mca.col_topandas())
        # Mapping des points lignes
        my_mca.mapping_row(num_x_axis=1, num_y_axis=2)
        # Classement des points lignes en fonction de leur contribution au 1er axe
        # Le paramètre de la methode plot_row_contrib indique que c'est pour l'axe numero 1 que les contributions sont ici 
        # representees
        my_mca.plot_row_contrib(num_axis=1)
        # Classement des points lignes en fonction de leur cos2 sur le 1er axe
        my_mca.plot_row_cos2(num_axis=1)
        # Mapping des points colonnes
        my_mca.mapping_col(num_x_axis=1, num_y_axis=2)
        # Classement des points colonnes en fonction de leur contribution au 1er axe
        my_mca.plot_col_contrib(num_axis=1)
        # Classement des points colonnes en fonction de leur cos2 sur le 1er axe
        my_mca.plot_col_cos2(num_axis=1)
        #graphes simultanees
        my_mca.mapping(num_x_axis=1, num_y_axis=2, short_labels=False,figsize=(10, 8))
        
class FAMDP(object):
    def __init__(self, famd,df_cible):
        
        self.famd=famd
        self.df = df_cible
    
    def plot_combined(self, labels=None):
        coords_indiv = self.famd.row_coordinates(self.famd.M_)
        coords_var = self.famd.column_coordinates(self.famd.M_)

        fig = go.Figure()

        # Ajout des individus
        fig.add_trace(go.Scatter(x=coords_indiv[:, 0], y=coords_indiv[:, 1], mode='markers',
                                  name='Individus', marker=dict(size=7, color='red', opacity=0.7)))

        # Ajout des variables
        fig.add_trace(go.Scatter(x=coords_var[:, 0], y=coords_var[:, 1], mode='markers',
                                  name='Variables', marker=dict(size=10, symbol='x', color='blue', opacity=0.7)))

        # Ajout des labels aux points, s'il y en a
        if labels:
            fig.add_trace(go.Scatter(x=coords_indiv[:, 0], y=coords_indiv[:, 1], mode='text',
                                      text=labels['Individus'], textposition='bottom center'))
            fig.add_trace(go.Scatter(x=coords_var[:, 0], y=coords_var[:, 1], mode='text',
                                      text=labels['Variables'], textposition='bottom center'))

        # Configuration du layout
        fig.update_layout(xaxis=dict(title='Dim1'), yaxis=dict(title='Dim2'),
                          title='Graphique des individus et des variables')

        return fig


    
class famd_prince(reduction_tools):
    def __init__(self,nb_composante=2,nb_iter=3,meth="famd",df=pd.DataFrame()):
        """

        Parameters
        ----------
        nb_composante : INT, optional
        meth_choice : 'famd' ou 'MFA'. quand la variable qualitative a 2 modalite, prefere la famd, sinon
                      il est preferable d'utiliser la meth MFA
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        reduction_tools.__init__(self,meth_choice=meth,df_cible=df,Y=pd.DataFrame())
        self.nb_compo = nb_composante
        self.nb_iter = nb_iter
        self.X = df
    def get_coord(self,col_label_name=""):
        import prince
        df = self.df_cible.copy()
        # Selectionner les colonnes categorielles
        col_categoric = df.select_dtypes(include=['object']).columns
        
        # Selectionner les colonnes numeriques
        col_numeric = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Remplacer les valeurs manquantes par la mediane des valeurs numeriques
        df[col_numeric] = df[col_numeric].fillna(df[col_numeric].median())
        
        # Remplacer les valeurs manquantes par "inconnu" pour les colonnes categorielles
        df[col_categoric] = df[col_categoric].fillna("inconnu")
        
        # Initialiser l'AFDM avec le nombre de dimensions souhaite
        if self.meth == 'MFA':
            print("============== methode MFA choisi: =============")
            af = prince.MFA(n_components=2,groups={"categorical":col_categoric})
            # Ajuster l'AFDM sur les donnees mixtes
            af = af.fit(df)
            # Recuperer les coordonnees des lignes dans l'espace factoriel
            coordinates = af.row_coordinates(df)
            # Afficher les resultats
            print(coordinates.head())
            plt.scatter(coordinates[0], coordinates[1])
            plt.xlabel('Axe factoriel 1 (' + str(round(af.explained_inertia_[0]*100, 2)) + '% d\'inertie)')
            plt.ylabel('Axe factoriel 2 (' + str(round(af.explained_inertia_[1]*100, 2)) + '% d\'inertie)')
            plt.title('Nuage de points des coordonnees des lignes dans l\'espace factoriel')
            plt.show()
        else:
            print("============== methode FAMD choisi: =============")
            af = prince.FAMD(n_components=self.nb_compo,n_iter=self.nb_iter,random_state=101)
            af = af.fit(df)
            # Ajuster l'AFDM sur les donnees mixtes
            af.plot_row_coordinates(df,figsize=(15, 10),color_labels=['{} : {}'.format(col_label_name,t) for t in df[col_label_name]] )
            # Recuperer les coordonnees des lignes dans l'espace factoriel
            coordinates = af.row_coordinates(df)
            # Afficher les resultats
            print(coordinates.head())
            return af
            
    def plot_mfa_3d(df):
        import prince
        # Realiser une analyse factorielle multiple
        mfa = prince.MFA(n_components=3, n_iter=3, copy=True, check_input=True, engine='auto', random_state=101)
        mfa.fit(df)
    
        # Recuperer les coordonnees des individus dans l'espace factoriel
        coordinates = mfa.row_coordinates(df)
    
        # Creer la figure 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=coordinates.iloc[:, 0],
            y=coordinates.iloc[:, 1],
            z=coordinates.iloc[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=coordinates.iloc[:, 2],  # Colorier selon la troisième composante
                colorscale='Viridis',           # Utiliser l'echelle de couleurs Viridis
                opacity=0.8
            )
        )])
    
        # Ajouter les titres et les etiquettes des axes
        fig.update_layout(scene=dict(
            xaxis_title='Axe factoriel 1 (' + str(round(mfa.explained_inertia_[0]*100, 2)) + '% d\'inertie)',
            yaxis_title='Axe factoriel 2 (' + str(round(mfa.explained_inertia_[1]*100, 2)) + '% d\'inertie)',
            zaxis_title='Axe factoriel 3 (' + str(round(mfa.explained_inertia_[2]*100, 2)) + '% d\'inertie)'),
            title='Visualisation en 3D des individus dans l\'espace factoriel')
    
        # Afficher la figure
        fig.show()
        
        
    
class Clustering(object):
    "developper une classe qui prend les resultat d'un acp, acm,fadm pour faire une classification grossiere avec un kmeans et ensuite une classification par arrbre"
    "cree une classe herite de reduction_tools pour pouvoir utiliser directement les fonctions et attributs"
        
def main(meth='pca'):    
    
     
    path = r'C:\Users\karl\Documents\datas\breast_cancer.csv'
    i_input = input(path,index="id_sample",separateur_csv=";")
    i_preprocess = preprocessing(i_input.data,col_label="pam50")
    X,Y = i_preprocess.if_labels()
    X_scaled = i_preprocess.centree_reduite(X)
    print("columns X:")
    print(X.head(5))
    print("columns Y:")
    print(Y.head(5))
    if meth=="pca":
        i_preprocess.ana_dist_labels()
        i_preprocess.ana_dist_numerics()
        X_scaled_woo = i_preprocess.get_rid_outliers(X_scaled)
        i_reduc = reduction_tools(df_cible=X_scaled,meth_choice=meth,Y=Y)
        X_pca = i_reduc.acp()
        i_reduc.plot_acp_result(nb_comp=3,b_3d_plot=True)
        
    elif meth =="pca_full":
        i_reduc = reduction_tools(df_cible=X_scaled,meth_choice=meth,Y=Y)
    elif meth == "tsne":
        i_reduc = reduction_tools(df_cible=X_scaled,meth_choice=meth,Y=Y)
        i_reduc.plot_tsne_result(nb_comp=3,nb_voisin=50)
    elif meth == "umap":
        i_reduc = reduction_tools(df_cible=X_scaled,meth_choice=meth,Y=Y)
        i_reduc.plot_umap_result(nb_comp=3)
    
        
    
if __name__=="__main__":
    start_time = datetime.now()
    #main(meth="umap")
    end_time = datetime.now()
    print("duree du script: {}".format(end_time-start_time))