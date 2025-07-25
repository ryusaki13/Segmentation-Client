
# Importation des librairies

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler,  OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Chargement du dataset à retrouver sur : "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
# Via Kaggle : "https://www.kaggle.com/code/aslibaraf/customer-behavior-analysis/" 

current_directory = os.getcwd()
file_path = os.path.join(current_directory,"Online Retail.csv")

df = pd.read_csv(file_path)

################# Exploration de la base de données ###################
print(df)
df.info()

df["Country"].value_counts()
df["Description"].nunique()

clients_total = df['CustomerID'].nunique()
print("Le nombre de clients est :",clients_total)

df["Montant"] = df["Quantity"]*df["UnitPrice"]
print(df["Montant"].head())
chiffre_affaire = df["Montant"].sum()
print('Le chiffre d\'affaire global est :', round(chiffre_affaire,0), "USD")

##################### DATA PREPROCESSING #########################
#Imputation des produit manquant (Description) et suppression de colonnes
df.isnull().sum()
df["Description"].mode()[0]
mode = df["Description"].mode()[0]
df["Description"].fillna(mode, inplace = True)
df.isnull().sum()

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceDate'].dtypes
df.info()

# Suppression des variables inutiles
df=df.drop(["InvoiceNo", "StockCode","Quantity","UnitPrice"],axis = 1)
df.head(10)
df.info()
df.isna().sum()
#########################################################################################
# Création d'un panier d'achat pour chaque customer
df[df['Montant'] <0]

df_panier =df.groupby(["CustomerID","InvoiceDate"], as_index= False )["Montant"].sum()
df_panier
col = {'Montant' : "Valeur_panier"}
df_panier.rename(columns =col, inplace = True)
df_panier
df_client = df_panier.groupby(["CustomerID"], as_index=False)["Valeur_panier"].sum()
df_client
valeur_moyenne = df_panier.groupby(["CustomerID"], as_index=False)["Valeur_panier"].mean()
panier_moyen =valeur_moyenne.iloc[:,1]
df_client["Valeur_moyenne"] = panier_moyen
df_client.head(5)
df_client.rename(columns={'Valeur_panier':"Valeur_totale"}, inplace = True)
########################################################################################
client_positif = df_client["Valeur_totale"] > 0
client_negatif = df_client[~client_positif]
len(client_negatif)
df_client_cleaned = df_client[client_positif]
#########################################################################################
file_path = os.path.join(current_directory, "df_client_cleaned.csv")
df_client_cleaned.to_csv(file_path, index = False)


######## NOUVEAU DATA PREPROCESSING ###################
import gc
gc.collect()

file_path_new = os.path.join(current_directory,"Online Retail.csv")
df = pd.read_csv(file_path_new)
df.head()

df_panier = df.groupby(["CustomerID","InvoiceDate"], as_index=False)["Montant_panier"].sum()
(df_panier["CustomerID"] == 99999).sum()
print(df_panier)

# Vérification des doublons
doublons = df_panier.duplicated().sum()
print(f"Il y a : {doublons} dans la base de données")


#######################         FEATURES ENGINEERING          #######################
###        CREATION DE NOUVELLE VARIABLES UTILIES POUR LA SEGMENTATION            ###
###                                  RFM ANALYSIS                                 ###
### Frequence: Le nombre de fois que le client à éffectué un achat                ###
### Recence: Le nombre de nombre de jour écoulés depuis le dernier achat          ###           
### Valeur : Le montant total dépensé par le client                               ### 
###                                                                               ### 
###            On ajoute uniquement de variable interpretable comme KPI           ### 
#####################################################################################
 
# On considère la dernière date d'achat comme date l'actuel .reset_index()

# Scoring Recence
date_actuelle = df_panier["InvoiceDate"].max()
df_panier["dernier_achat"] = (date_actuelle - df_panier["InvoiceDate"]).dt.days
dernier_achat_moyen = df_panier.groupby(['CustomerID'])["dernier_achat"].min().reset_index()
dernier_achat_moyen.columns=["CustomerID","Recence"]
dernier_achat_moyen["Score_recence"] = np.log1p(dernier_achat_moyen["Recence"])
dernier_achat_moyen

# Scoring Retention
premier_achat_moyen = df_panier.groupby("CustomerID")['dernier_achat'].max().reset_index()
premier_achat_moyen.columns =["CustomerID","Retention"]
premier_achat_moyen["Score_retention"] = np.log1p(premier_achat_moyen["Retention"])
premier_achat_moyen
dernier_achat_moyen

dernier_achat_moyen = dernier_achat_moyen.merge(premier_achat_moyen, on ='CustomerID')
dernier_achat_moyen

# Scoring dépense : Montant total dépensé

panier_total = df_panier.groupby(["CustomerID"])["Valeur_panier"].sum().reset_index()
panier_total.columns = ["CustomerID","Depense"]
panier_total["Score_depense"] = np.log1p(panier_total["Depense"])
dernier_achat_moyen = dernier_achat_moyen.merge(panier_total, on = "CustomerID")
dernier_achat_moyen

# Scoring Frequence
sns.boxplot(dernier_achat_moyen["Depense"])
plt.show()
nombre_commande = df_panier.groupby(["CustomerID"])["InvoiceDate"].nunique().reset_index()
nombre_commande.columns = ["CustomerID","Frequence"]
nombre_commande["Score_frequence"] = np.log1p(nombre_commande["Frequence"])
dernier_achat_moyen = dernier_achat_moyen.merge(nombre_commande, on = "CustomerID")
dernier_achat_moyen

RFM = dernier_achat_moyen.copy()
RFM.head(5)

# Customer Lifetime value : Valeur Vie Client

nommbre_panier = df_panier.groupby(["CustomerID"])["InvoiceDate"].count().reset_index()
nommbre_panier.columns = ["CustomerID","Nombre_commande"]
df_panier = df_panier.merge(nommbre_panier, on = "CustomerID")
panier_total = panier_total.merge(nommbre_panier, on = "CustomerID")
panier_total["Depense_moyenne"] = panier_total['Depense'] / panier_total['Nombre_commande']

panier_total["CLV"] = panier_total["Depense_moyenne"]*panier_total["Nombre_commande"]*(RFM["Retention"]/360)
panier_total
# Pas très optimal, peut etre calculer directement avec la base RFM
RFM["Score_RFM"] = RFM['Recence']*RFM['Frequence']*RFM['Depense']
RFM['CLV'] = panier_total['CLV']
RFM.head()

# Ration de dépenses fréquente

RFM["Ratio_depense_frequence"] = RFM["Frequence"] / (RFM["Recence"]+ 1)
seuil_monetaire = RFM["Depense"].quantile(0.75)
seuil_frequence = RFM["Frequence"].quantile(0.75)
RFM["Client_haute_valeur"] = (RFM["Depense"]> seuil_monetaire) & (RFM["Frequence"] > seuil_frequence)
RFM.head(15)
seuil_monetaire
seuil_frequence
RFM["Ratio_depense_frequence"].describe()
Nombre_client_haute_valeur = RFM["Client_haute_valeur"].sum()
print(f"Nous avons {Nombre_client_haute_valeur}, qui peuvent etre considérer comme de VIP car ils ont dépensés plus\n de {seuil_monetaire} USD avec plus de {seuil_frequence} achats")
RFM.columns
RFM.isnull().sum()
RFM = RFM.dropna()


# Régression polynomiale et recherche d'un intervalle de prédiction 

def intervalle_prediction(x, y, percentile = 0.99) :
    model = np.poly1d(np.polyfit(x,y,3))
    x_minmax = np.linspace(x.min(), x.max(), 100)
    y_pred = model(x_minmax)
    se = np.sqrt((np.sum((y - model(x))**2)) / (len(y) - 2))
    t_value = np.abs(np.percentile(np.random.standard_t(df=len(x)-2, size=1000), 100 * (1 - (1 - percentile) / 2)))
    bound = t_value * se * np.sqrt(1 + 1/len(y) + (x_minmax - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

    return model, x_minmax, y_pred, bound
    
# Function to remove outliers based on prediction bounds
def supprimer_outlier(ddf, x_col, y_col):
    x = ddf[x_col]
    y = ddf[y_col]
    model, x_minmax, y_pred, bound = intervalle_prediction(x, y)
    predictions = model(x)
    se = np.sqrt((np.sum((y - model(x))**2)) / (len(y) - 2))
    t_value = np.abs(np.percentile(np.random.standard_t(df=len(x)-2, size=1000), 100 * (1 - (1 - 0.99) / 2)))
    bound = t_value * se * np.sqrt(1 + 1/len(y) + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    
    lower_bound = predictions - bound
    upper_bound = predictions + bound
    
    inliers = (y >= lower_bound) & (y <= upper_bound)
    return ddf[inliers]

colonnes_requis = ['Score_recence', 'Score_depense', 'Score_frequence']
data = RFM[colonnes_requis]

pairs = [('Score_recence', 'Score_depense'), ('Score_frequence','Score_depense')]
cleaned_data = {}

for x_col, y_col in pairs:
    cleaned_data[(x_col, y_col)] = supprimer_outlier(data, x_col, y_col)

# Generate the regression plots for each pair
for x_col, y_col in pairs:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_col, y=y_col, data=cleaned_data[(x_col, y_col)])
    model, x_minmax, y_pred, bound = intervalle_prediction(cleaned_data[(x_col, y_col)][x_col], cleaned_data[(x_col, y_col)][y_col])
    plt.plot(x_minmax, y_pred, color='red')
    plt.fill_between(x_minmax, y_pred - bound, y_pred + bound, color='red', alpha=0.2)
    plt.xlabel(x_col) 
    plt.ylabel(y_col) 
    plt.title(f'Regression Plot of {x_col} vs {y_col} with Prediction Bounds')
    plt.show(block = False)
    plt.pause(2)

# Analyse de corrélation entre les variables

matrice = RFM[["Score_recence","Score_retention","Score_depense","Score_frequence","CLV","Client_haute_valeur"]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(matrice, annot = True, cmap = 'coolwarm', linewidths = 0.5, vmin = -1, vmax=1)
plt.title("Matrice de corrélation RFM")
plt.xticks(rotation=45, ha='right')
plt.show(block = False)

file_path_RFM = os.path.join(current_directory, "CustomerRFM.csv")
RFM.to_csv(file_path_RFM, index = False )

##########################################################################################################################################################################

##################    ANALYSE EN COMPOSANTE PRINCIPALE (ACP)       ##################
###                                                                               ###
###   L'ACP aide  réduire la dimension et regrouper les clients selons leurs      ###
###   Caractaristiques communes                                                   ### 
#####################################################################################


import gc
gc.collect()
print("Mémoire vidé")



RFM = pd.read_csv(file_path_RFM)
RFM.head(5)

# ACP avec base centré réduite
df_pca = RFM[["Score_recence","Score_retention","Score_depense","Score_frequence","Client_haute_valeur"]]
scaler = StandardScaler()
df_centre_reduit = scaler.fit_transform(df_pca)

# Modele ACP
pca = PCA()
pca.fit_transform(df_centre_reduit)
variance_expliquee = pca.explained_variance_ratio_
valeur_propre = pca.explained_variance_

# Analyse des valeurs propre  : Variance expliquée
table_pca = pd.DataFrame({
    "Dimension" : ["Dim"+str(i +1) for i in range(pca.n_components_)],
    "Valeur_propre" : pca.explained_variance_,
    "%_Variance_expliquee" : np.round(variance_expliquee*100, 2),
     "%_Cumul_variance_expliquee" : np.round(np.cumsum(variance_expliquee*100),2)
     })
table_pca
sns.barplot(
    x = "Dimension",
    y="%_Variance_expliquee",
    data = table_pca
    )
plt.plot(table_pca["Dimension"],table_pca["%_Variance_expliquee"],marker='o', linestyle='--', color = "red")
plt.title("Pourcentage de variance expliquee par dimension")
plt.show(block =False)

plt.figure(figsize =(10,5))
plt.plot(table_pca["Dimension"],table_pca["%_Variance_expliquee"],marker='o', linestyle='--', color = "red")
plt.title("Eboulis de valeur propre")
plt.ylabel("Pourcentage de variance expliquee")
plt.xlabel("Dimension")
plt.show()

# Qualite de representation des variables
cos = pca.components_.T #loading ou correlation entre les variables et les 
colonne = ["Axe" + str(i+1) for i in range(pca.n_components_)]
loadings = pd.DataFrame(cos, columns = colonne, index = df_pca.columns)
loadings
sns.heatmap(loadings, annot = True, cmap ="YlGnBu")
plt.title("Matrice de correlation")
plt.show()

contribution_variable = (cos**2)*valeur_propre  #la contribution des variables à la formation des axes
table_contribution = pd.DataFrame(contribution_variable, columns= colonne, index = df_pca.columns )
table_contribution
# Axe 1 est formé essentiellement par les clients les plus fréquents, qui dépensent assez (Score_depense +++ Score_frequence +++ Client_haute_valeur ++++)
# axe 2 est formé par les clients anciens et qui ont achété récemment (Score_recence ++ Score_retention +++ )

# Matrice de contribution des variables à la formation des axes
sns.heatmap(table_contribution, annot = True,cmap='magma')
plt.title("Matrice de contribution")
plt.show()
##########################################################################################################################################################################

##################               CLUSTERING AVEC K-MEANS           ##################
###                                                                               ###
###   Les clients seront regroupeés dans des clusters ou grouypes selons leur     ###
###   Caractaristiques communes, chaque cluster represent un ensemble de clients  ### 
###   ayants des comportement plus ou moins semblables.                           ### 
#####################################################################################

df_kmeans = RFM[["Score_recence","Score_retention","Score_depense","Score_frequence","Client_haute_valeur"]].copy()
df_kmeans


# Nombre optimal de cluster avec les inerties(somme des distances quadratiques entre chauqe points du cluster et le centroid)
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=19)
    kmeans.fit(df_kmeans)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1,11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Autre technique pour derminer les nombre optimal de cluster toujours avec la méthode du coude
from yellowbrick.cluster import KElbowVisualizer

df_kmeans = RFM[["Score_recence","Score_retention","Score_depense","Score_frequence","Client_haute_valeur"]].copy()
kmeans = KMeans()
graph = KElbowVisualizer(kmeans, k=(1,11))
graph.fit(df_kmeans)
graph.show()

# Le nombre optimal de cluster est 4 pour les deux techniques

kmeans = KMeans(n_clusters= 4, random_state= 19)
kmeans.fit(df_kmeans)
df_kmeans["Segment"] = kmeans.fit_predict(df_kmeans)
df_kmeans.head(15)

df_kmeans["Segment"].value_counts()
df_kmeans["Client_haute_valeur"].sum() # ils sont tous dans le cluster 1, des clients 

##### Clustering avec les données de l'ACP

# ACP avec base centré réduite
df_pca = RFM[["Score_recence","Score_retention","Score_depense","Score_frequence","Client_haute_valeur"]]
scaler = StandardScaler()
df_centre_reduit = scaler.fit_transform(df_pca)

# Modele ACP
pca = PCA(n_components= 3) # On choisir de se limité à 3 facteurs car ils expliquent plus de 91% de la variance
df_facteurs = pca.fit_transform(df_centre_reduit)

valeur_propre = pca.explained_variance_
cos = pca.components_.T
contrib = (cos**2)*valeur_propre
contrib_pourcentage = contrib * 100
colonne = ["Axe"+str(x+1) for x in range(pca.n_components_)]
tableau_contrib = pd.DataFrame(contrib_pourcentage, columns = colonne, index= df_pca.columns)

# on cherche le nombre optimal de cluster avec les donnees de l'ACP
kmeans = KMeans()
graph = KElbowVisualizer(kmeans, k = (1, 11))
graph.fit(df_facteurs)
graph.show()
# On a également 4 clusters, le nombre de cluster n'est pas impacté selon qu'on utilise les données de l'ACP ou non

kmeans = KMeans(n_clusters= 4, random_state=19)
kmeans.fit(df_facteurs)
df_kmeans["Cluster_acp"] = kmeans.labels_
df_kmeans.head(15)

df_kmeans["Segment"].value_counts()
df_kmeans["Cluster"].value_counts()
df_kmeans["Client_haute_valeur"].value_counts()
df_kmeans

sns.heatmap(tableau_contrib, annot= True, cmap = "magma")
plt.title("Matrice de contribution des variables à la formation des axes")
plt.ylabel("Contribution en %")
plt.xlabel("Axes principaux")
plt.show(block = False)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_facteurs[:, 0], y=df_facteurs[:, 1], hue= df_kmeans['Cluster_acp'],  palette='magma')
plt.title('Clusters avec donnes ACP')
plt.xlabel('Facteur (Axes) 1')
plt.ylabel('Facteur (Axes) 2')
plt.legend(title='Cluster ACP')
plt.show(block = False)

plt.figure(figsize=(10, 5))
sns.scatterplot(data=RFM, x='Score_recence', y='Score_frequence', hue='Cluster_acp', palette='viridis')
plt.title('K-means Clustering')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(data=RFM, x='Score_frequence', y='Score_depense', hue='Cluster_acp', palette='viridis')
plt.title('K-means Clustering')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(data=RFM, x='Score_recence', y='Score_depense', hue='Cluster_acp', palette='viridis')
plt.title('K-means Clustering')
plt.show(block = False)

plt.figure(figsize=(10, 5))
sns.scatterplot(data=RFM, x='Score_frequence', y='Score_recence', hue='Cluster_acp', palette='viridis')
plt.title('K-means Clustering')
plt.show(block = False)

silhouette = silhouette_score(df_facteurs, df_kmeans['Cluster_acp'] )
db_indice = davies_bouldin_score(df_facteurs, df_kmeans['Cluster_acp'])
ch_indice = calinski_harabasz_score(df_facteurs, df_kmeans['Cluster_acp'])
print(f"silhouette_score : {silhouette:.3f} ")
print(f"davies_bouldin_score : {db_indice:.3f} ")
print(f"calinski_harabasz_score : {ch_indice:.3f} ")

centroids = pd.DataFrame(df_kmeans.groupby(["Cluster_acp"])[["Score_recence","Score_retention","Score_depense","Score_frequence","Client_haute_valeur"]].mean())
centroids

df_kmeans = df_kmeans.drop(["Segment"], axis = 1)
df_kmeans
df_kmeans["CustomerID"] = RFM["CustomerID"]
RFM = RFM.merge(df_kmeans[["CustomerID","Cluster_acp"]], on = "CustomerID", how = "left")
RFM[RFM["Cluster_acp"]==1] # cluster 1 : client VIP, anciens très fréquents et récens avec un haut pouvoir d'achat

# Cluster 3 : Clients très anciens (haute score de rentention) et très récents (ils ont achété récement) avec un pouvoir d'achat relativement mobasyen.
# Ils ont achété récement, ils ne sont pas fréquent avec un score de fréquence = 0.95 qui se situe entre le Q1 et Q2 : [0.69 , 1.38]

RFM[RFM["Cluster_acp"]==3]
RFM["Score_retention"].describe()
RFM["Depense"].describe()
RFM["Score_frequence"].describe()
RFM[RFM["Cluster_acp"]==3]["Score_frequence"].mean()


# Cluster 2 : groupe de clients venant juste le groupe 1, ils récent avec un pouvoir d'achat relativement solide c'est une sorte de classe moyenne
# Ils plus fréquents que le cluster 3 et moins que le premier, c'est les deuxièmes  après les VIP

RFM[RFM["Cluster_acp"]==2]["Score_frequence"].mean()


RFM[RFM["Cluster_acp"]==0]["Depense"].mean()
RFM["Depense"].mean()

# Cluster 0 : clients très ancien et qui ont achétés récement mais faible pouvoir d'achat faible fréquence et pourvoir d'achat 

description = {
"Cluster" : ["Cluster 0 : Inactifs","Cluster 1 : Actifs de classe moyenne","Cluster 2 : VIP & Champions","Cluster 3 : Anciens mais peu engages"],
"Description_segment" : ["Clients qui ont pas achete depuis longtemps\n faible frequence et faible montant depense",
                         "Meilleurs clients: client depuis longtemps\n depensent beaucoup et achetent tres frequemment",
                         "Achat regulier et montant moyen\n fréquence et récence correctes mais possibilite de montee en gamme",
                         "Clients qui ont achete tres recemment\n mais avec faible frequence et montant modeste\n reviennent aprs une longue pause"],
"Actions_marketing_ commerciales" : ["Campagne de reactivation\n Relance personnalisee",
                                     "Programme premium\n invitations a des ventes privaes\n Fidalisation renforcee\n service client dedie",
                                     "Crossse-lling / Up-selling\n Offre fidelite graduee\n suggestions de produits complementaires ou premium",
                                     "programme de parrainage\n Incitation a la repetition\n Un feedback recompense (analyse de la satisfaction)"]
}

df_marketing = pd.DataFrame(description)
df_marketing

#########################################################################################################
# EXPORTATION DU MODELE ET PIPELINE

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 


RFM = pd.read_csv(file_path_RFM)
colonne = ["Score_recence","Score_retention","Score_depense","Score_frequence","Client_haute_valeur"]
base_entrainement = RFM[colonne].copy()

# Création du tranformer qui prepare nos donnees et applique l'ACP
class AnalyseFactorielle(BaseEstimator, TransformerMixin) :
    def __init__(self, nombre_facteur):
        self.centre_reduis = StandardScaler()
        self.nombre_facteur = nombre_facteur
        self.pca = PCA(n_components=nombre_facteur)
        
    
    def fit(self, X, y= None):
        X_centre_reduis = self.centre_reduis.fit_transform(X)
        self.pca.fit(X_centre_reduis)
        return self
    
    def transform(self,X, y =None):
        X_centre_reduis = self.centre_reduis.transform(X)
        X_pca = self.pca.transform(X_centre_reduis)
        return X_pca
    

# Creation d'une pipeline
pipeline = Pipeline([
    ("Modele_acp",AnalyseFactorielle(nombre_facteur = 3)),
    ("Kmeans", KMeans(n_clusters= 3, random_state= 19))
])

# Entrainement de la pipeline
pipeline.fit(base_entrainement)

# Enregistrement de la pipeline
joblib.dump(pipeline, 'mon_pipeline_segmentation.joblib')

#########################################################################
# Test du pipeline et du modele

import joblib

# Chargement
pipeline_charge = joblib.load('mon_pipeline_segmentation.joblib')

def traitement_et_prediction(nouvelle_base) :
    return pipeline_charge.predict(nouvelle_base)  

nouvelle_base = pd.DataFrame({
    "Score_recence": [8.5,3,9.2],
    "Score_retention": [5,2.4,2],
    "Score_depense": [6,3,9],
    "Score_frequence":[8,5,2],
    "Client_haute_valeur":[1,0,0]
})

base_exemple = nouvelle_base.copy()
base_exemple["Cluster"] = traitement_et_prediction(base_exemple)


#################### FIN DU PROJET #########################################
















