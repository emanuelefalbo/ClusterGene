#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("white")
from sklearn.impute import KNNImputer
# from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split
import argparse
# import lightgbm as lgbm

plt.rcParams['figure.figsize'] = [12, 6]

def cml_parser():
    parser = argparse.ArgumentParser('EnGene.py', formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('option_file', nargs="?", help="json option file")
    parser.add_argument('filename', nargs="?",  help="input file")
    parser.add_argument('-n', default="impute", choices=["drop", "impute"], help="choose to drop Nan or impute")
    parser.add_argument('-t', default="medoids", choices=["centroids", "medoids"], help="choose between KMeans and K-medoids clustering algorithms")

    opts = parser.parse_args()
    if opts.filename == None:
        raise FileNotFoundError('Missing input file or None')
    
    return  opts

def drop_na(df):
    print(f" Existing Missing values: dropping NaN ...")
    df_dropped = df.dropna()
    
    return df_dropped

def impute_data(df):
    # Imputing data by means of K-Nearest Neighbours algo
    print(f" Existing Missing values: Imputing data ...")
    knn_imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean')
    df_knn = df.copy()
    df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)
    columns_Nans = df_knn.columns[df_knn.isna().any()].to_list()
    null_values = df_knn[columns_Nans[0]].isnull()
    fig = plt.figure()
    fig = df_knn_imputed.plot(x=df_knn.columns[0], y=columns_Nans[0], kind='scatter', c=null_values, cmap='winter', 
                         title='KNN Imputation', colorbar=False, edgecolor='k', figsize=(10,8))
    # plt.legend()
    plt.savefig("KNN_imputed_column_0th.png")

    return df_knn_imputed

class DoClusters():
    def __init__(self, X, n_clusters, mode):
        self.X = X
        self.kmin = 2
        self.kmax = n_clusters+1  # include last 
        self.mode = mode

    def do_clusters(self):
        if self.mode == "centroids":
            from sklearn.cluster import KMeans
            print(f"Clustering by KMeans .... ")
            
            model = [ KMeans(k, n_init=10).fit(self.X) for k in range(self.kmin, self.kmax) ]
            
        elif self.mode == "medoids":
            from sklearn_extra.cluster import KMedoids
            print(f"Clustering by K-medoids .... ")
            
            
            model = [ KMedoids(k, n_init=10).fit(self.X) for k in range(self.kmin, self.kmax) ]

        self.inert_ = [ model[k].inertia_ for k in range(len(model))]
        self.silho_ = [ silhouette_score(self.X, model[k].labels_) for k in range(len(model))]
        self.ch_ = [  calinski_harabasz_score(self.X, model[k].labels_) for k in range(len(model))]
        self.db_ = [  davies_bouldin_score(self.X, model[k].labels_) for k in range(len(model))]
        
        
        # Find best no cluster from the 3 out 4 tests:
        # if NaN present round the mean of knee locators of each score
        x = range(self.kmin, self.kmax)
        print(len(x), len(self.ch_))
        best_scores = []
        for idx, score in enumerate([self.inert_, self.silho_, self.ch_, self.db_]):
            if idx == 0 or idx == 2:
                best_scores.append(KneeLocator(x, score, curve="convex", direction="decreasing").knee)
            elif idx == 1 or idx == 3:
                best_scores.append(KneeLocator(x, score, curve="concave", direction="increasing").knee)

        best_scores = np.array(best_scores, dtype=float)
        best_scores = best_scores[~np.isnan(best_scores)]   # Remove NaN
        best_knee = np.round(best_scores.mean())
        print(best_knee)
        
        return best_knee
        
        
        # return model, self.inert_, self.silho_, self.ch_, self.db_
    
    # def display(self):
    #     fig, ax = plt.subplots(2,2)
    #     ax[0,0].plot(np.arange(2, len(self.inert_)+2), self.inert_, '-o')
    #     ax[0,0].set_title("Inertia Score")
    #     ax[0,1].plot(np.arange(2,len(self.silho_)+2), self.silho_, '-o')
    #     ax[0,1].set_title("Silhouette Score")
    #     ax[1,0].plot(np.arange(2, len(self.ch_)+2), self.ch_, '-o')
    #     ax[1,0].set_title("CH Score")
    #     ax[1,1].plot(np.arange(2, len(self.db_)+2), self.db_, '-o')
    #     ax[1,1].set_title("DB Score")
    #     plt.tight_layout()
    #     plt.savefig('Model_Scores.png', dpi=100,  bbox_inches='tight', pad_inches=0.1)

        
def main():
    opts = cml_parser()
    sep = None
    if opts.filename[-3:] == "tsv":
        sep == "\t"
    df = pd.read_csv(opts.filename, sep=sep, engine="python")
    # Drop Nan or Impute data
    if opts.n == "drop":
        df = drop_na(df)
    elif opts.n == "impute":
        df = impute_data(df)
        
    # DoClusters class takes X(n_sample, n_features) array
    # and the no of clusters to perform clustering according the opt.t mode:
    # drop NaN or impute data
    model = DoClusters(X=df.to_numpy(), n_clusters=4, mode=opts.t)    # you might let the user choose the n_clusters
    model.do_clusters()


    

    

if __name__ == "__main__":
    main()

