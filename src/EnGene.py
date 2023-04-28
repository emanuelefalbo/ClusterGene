#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("white")
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split
import argparse
# import lightgbm as lgbm

plt.rcParams['figure.figsize'] = [12, 6]


import warnings
warnings.filterwarnings('ignore')

def range1(start, end):
    return range(start, end+1)

def cml_parser():
    parser = argparse.ArgumentParser('EnGene.py', formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('option_file', nargs="?", help="json option file")
    # parser.add_argument('filename', nargs="?",  help="input file")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input file CRISPR-Cas9 matrix', required=True)
    requiredNamed.add_argument('-ref', '--reference', help='Input reference file name', required=True)
    requiredNamed.add_argument('-t', '--tissue', help='Input tissue to be parsed', required=True)
    parser.add_argument('-m', default="impute", choices=["drop", "impute"], help="choose to drop Nan or impute")
    parser.add_argument('-n', default=10, type=int, help="Number of clusters for  clustering algorithms")
    parser.add_argument('-k', default="centroids", choices=["centroids", "medoids", "both"], help="choose between KMeans and K-medoids clustering algorithms or both")
    opts = parser.parse_args()
    if opts.input == None:
        raise FileNotFoundError('Missing CRISPR-Cas9 input file or None')
    elif opts.reference == None:
         raise FileNotFoundError('Missing reference input file or None')
    return  opts

def drop_na(df):
    print(f" If existing missing values: dropping NaN ...")
    columns_Nans = df.columns[df.isna().any()].to_list()
    if len(columns_Nans) != 0:
        df_dropped = df.dropna()
    else:
        df_dropped = df
    return df_dropped

def impute_data(df):
    # Imputing data by means of K-Nearest Neighbours algo
    print(f" if existing missing values: Imputing data ...")
    knn_imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean')
    df_knn = df.copy()
    columns_Nans = df_knn.columns[df_knn.isna().any()].to_list()
    if len(columns_Nans) != 0:
       df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)
       null_values = df_knn[columns_Nans[0]].isnull()
       fig = plt.figure()
       fig = df_knn_imputed.plot(x=df_knn.columns[0], y=columns_Nans[0], kind='scatter', c=null_values, cmap='winter', 
                            title='KNN Imputation', colorbar=False, edgecolor='k', figsize=(10,8))
       # plt.legend()
       plt.savefig("KNN_imputed_column_0th.png")
    else:
        df_knn_imputed = df
    return df_knn_imputed

class DoClusters():
    def __init__(self, X, n_clusters, mode):
        self.X = X.to_numpy()     # Convert DF to Array
        self.kmin = 2
        self.kmax = n_clusters    # include last 
        self.mode = mode
        self.index = X.index      # Get index of DF

    def do_clusters(self):
        # if self.mode == "both":
        if self.mode == "centroids":
            print(f" Clustering by KMeans .... ")
            self.model = [ KMeans(k, n_init=10).fit(self.X) for k in range1(self.kmin, self.kmax) ]
            print(self.model)
        elif self.mode == "medoids":
            print(f" Clustering by K-medoids .... ")
            self.model = [ KMedoids(k).fit(self.X) for k in range1(self.kmin, self.kmax) ]
        return self.model

    def get_score_n_knees(self):
        self.inert_ = [ self.model[k].inertia_ for k in range(len(self.model))]
        self.silho_ = [ silhouette_score(self.X, self.model[k].labels_) for k in range(len(self.model))]
        self.ch_ = [  calinski_harabasz_score(self.X, self.model[k].labels_) for k in range(len(self.model))]
        self.db_ = [  davies_bouldin_score(self.X, self.model[k].labels_) for k in range(len(self.model))]
        # Find best no cluster from the 3 out 4 tests:
        # if NaN present round the mean of knee locators of each score
        x = range1(self.kmin, self.kmax)
        if len(self.model) >= 2:
            best_scores = []
            for idx, score in enumerate([self.inert_, self.silho_, self.ch_, self.db_]):
                if idx == 0 or idx == 2:
                    best_scores.append(KneeLocator(x, score, curve="convex", direction="decreasing").knee)
                elif idx == 1 or idx == 3:
                    best_scores.append(KneeLocator(x, score, curve="concave", direction="increasing").knee)
            best_scores = np.array(best_scores, dtype=float)
            best_scores = best_scores[~np.isnan(best_scores)]   # Remove NaN
            self.best_knee = int(np.round(best_scores.mean()))
            return best_scores, self.best_knee
        else:
            best_scores = None
            self.best_knee = None
        return best_scores, self.best_knee
    
    def display_score(self):
        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(np.arange(self.kmin, self.kmax), self.inert_, '-o')
        ax[0,0].set_title("Inertia Score")
        ax[0,1].plot(np.arange(self.kmin, self.kmax), self.silho_, '-o')
        ax[0,1].set_title("Silhouette Score")
        ax[1,0].plot(np.arange(self.kmin, self.kmax), self.ch_, '-o')
        ax[1,0].set_title("CH Score")
        ax[1,1].plot(np.arange(self.kmin, self.kmax), self.db_, '-o')
        ax[1,1].set_title("DB Score")
        plt.tight_layout()
        plt.savefig('Model_Scores.png', dpi=100,  bbox_inches='tight', pad_inches=0.1)
    
    def write_label_to_csv(self):
        # write the labels to CSV file
        if len(self.model) > 1:
            for idx, var in enumerate(self.model):
                    if var.get_params()["n_clusters"] == self.best_knee:
                        labels = pd.Series(var.labels_, index=self.index)
                        break
            labels.to_csv("Labels_from_clustering.csv")
        else:
            labels = pd.Series(self.model[0].labels_, index=self.index)
        labels.to_csv("Labels_from_clustering.csv")


def annote(df_map, df_cl, tissue):
    depmap_id = df_cl["depmapId"]
    lineage_1 = df_cl["lineage1"]
    lineage_1_unique = list(set(lineage_1))
    if tissue in lineage_1_unique:
        id_tissue = lineage_1[lineage_1 == tissue].index
        name_cl = depmap_id[id_tissue].to_list()
        #Parse DepMap to select above cell lines 
        id_true = []
        count = 0
        for k, var in enumerate(df_map.columns):
            if var in name_cl:
                id_true.append(k)
                count +=1
        df_tissue = df_map.iloc[:, id_true]
        df_tissue = df_tissue.add_prefix(f'{tissue} ')
        return df_tissue
    else:
        msg = ' \n'.join(lineage_1_unique)
        print(f' {tissue} not present in lineage1 \n')
        print(f' Select from the following list: \n')
        print(f' {msg} \n')


def main():
    opts = cml_parser()
    sep = None
    if opts.input[-3:] == "tsv":
        sep == "\t"
    df_map = pd.read_csv(opts.input, sep=sep, engine="python", index_col=0)
    df_cl = pd.read_csv(opts.reference, sep=sep, engine="python")
    if df_map.shape[0] < df_map.shape[1]:  # Reverse order : (Rows x Colums) = (Gene x Cell Lines) 
        df_map = df_map.T
    # Select no of Cell lines for specific tissue
    df_tissue = annote(df_map, df_cl, opts.tissue) 
    # # Drop Nan or Impute data
    if opts.m == "drop":
        df_tissue = drop_na(df_tissue)
    elif opts.m == "impute":
        df_tissue = impute_data(df_tissue)
    # DoClusters class takes X(n_sample, n_features) DataFrame and the no of clusters
    # to perform clustering according the opt.k mode
    clusters_ = DoClusters(X=df_tissue, n_clusters=opts.n, mode=opts.k) 
    model = clusters_.do_clusters()
    best_scores, best_knee = clusters_.get_score_n_knees()
    clusters_.write_label_to_csv()
    # print(best_scores, best_knee)



if __name__ == "__main__":
    main()

