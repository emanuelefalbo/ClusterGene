#!/usr/bin/env python3

import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("white")
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from sklearn.decomposition import PCA
import argparse
import os
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [12, 6]

def range1(start, end):
    return range(start, end+1)

def cml_parser():
    parser = argparse.ArgumentParser('EnGene.py', formatter_class=argparse.RawDescriptionHelpFormatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input file CRISPR-Cas9 matrix', required=True)
    requiredNamed.add_argument('-ref', '--reference', help='Input reference file name', required=True)
    requiredNamed.add_argument('-t', '--tissue',default='all', help='Input tissue to be parsed', required=True)
    parser.add_argument('-m', default="impute", choices=["drop", "impute"], help="choose to drop Nan or impute")
    parser.add_argument('-n', default=2, type=int, help="Number of clusters for  clustering algorithms")
    parser.add_argument('-k', default="centroids", choices=["centroids", "medoids", "both"], help="choose between KMeans and K-medoids clustering algorithms")
    opts = parser.parse_args()
    if opts.input == None:
        raise FileNotFoundError('Missing CRISPR-Cas9 input file or None')
    elif opts.reference == None:
         raise FileNotFoundError('Missing reference input file or None')
    return  opts


def remove_string(df):
    index = df.index.tolist()
    idx = []
    for i in index:
        if i.split()[1][0] == '(':
           idx.append(i.split('(', 1)[0].strip())
    # idx = [ i.split('(', 1)[0].strip() for i in index]
    df.reset_index(drop=True)
    df_mod = pd.DataFrame(df.to_numpy(), index=idx, columns=df.columns)
    
    return df_mod

def read_input(opts):
    sep = None
    if opts.input[-3:] == "tsv":
        sep == "\t"
        df_map = pd.read_csv(opts.input, sep=sep, engine="python", index_col=0)
        df_cl = pd.read_csv(opts.reference, sep=sep, engine="python")
    elif opts.input[-3:] == "csv":
        sep == ","
        df_map = pd.read_csv(opts.input, sep=sep, engine="python", index_col=0)
        df_cl = pd.read_csv(opts.reference, sep=sep, engine="python")
    elif opts.input[-3:] == "txt":
        df_map = pd.read_table(opts.input, engine="python", index_col=0)
    if df_map.shape[0] < df_map.shape[1]:  # Reverse order : (Rows x Colums) = (Gene x Cell Lines) 
        df_map = df_map.T
    df_mod = remove_string(df_map)  
    return df_mod ,df_cl

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
    print(f" if existing missing values: KNN-imputing data ...")
    knn_imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean')
    df_knn = df.copy()
    columns_Nans = df_knn.columns[df_knn.isna().any()].to_list()
    if len(columns_Nans) != 0:
       df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)
    #    null_values = df_knn[columns_Nans[0]].isnull()
    #    fig = plt.figure()
    #    fig = df_knn_imputed.plot(x=df_knn.columns[0], y=columns_Nans[0], kind='scatter', c=null_values, cmap='winter', 
    #                         title='KNN Imputation', colorbar=False, edgecolor='k', figsize=(10,8))
    #    # plt.legend()
    #    plt.savefig("KNN_imputed_column_0th.png")
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
            self.model = [ KMeans(k, n_init=10, random_state=0).fit(self.X) for k in range1(self.kmin, self.kmax) ]
        elif self.mode == "medoids":
            self.model = [ KMedoids(k).fit(self.X) for k in range1(self.kmin, self.kmax) ]
        return self.model

    def get_score_n_knees(self):
        self.inert_ = [ self.model[k].inertia_ for k in range(len(self.model))]
        self.silho_ = [ silhouette_score(self.X, self.model[k].labels_) for k in range(len(self.model))]
        self.ch_ = [  calinski_harabasz_score(self.X, self.model[k].labels_) for k in range(len(self.model))]
        self.db_ = [  davies_bouldin_score(self.X, self.model[k].labels_) for k in range(len(self.model))]
        # Find best no cluster from the 3 out 4 score tests:
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
            return  best_scores, self.best_knee
        else:
            best_scores = None
            self.best_knee = None
        return best_scores, self.best_knee
    
    def plot_score(self):
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
    
    def labels_to_csv(self, nameout):
        nameout =  nameout.replace('/', '_')
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        fout = os.path.join(out_dir, nameout + ".csv")
        # write the labels to CSV file
        if len(self.model) > 1:
            for idx, var in enumerate(self.model):
                    if var.get_params()["n_clusters"] == self.best_knee:
                        # labels = pd.Series(var.labels_, index=self.index)
                        labels = pd.DateFrame({'Gene': self.index[idx], 'label': var.labels_}, index=self.index)
                        break
            # labels.rename(columns={'0':'label'})
            labels.to_csv(fout)
        else:
            labels = pd.Series(self.model[0].labels_, index=self.index)
            # labels.rename(columns={'0':'label'})
        labels.to_csv(fout)
        return labels
    

        
def calculate_iqr(df, col_name, th1=0.25, th3=0.75):
    quartile1 = df[col_name].quantile(th1)
    quartile3 = df[col_name].quantile(th3)
    iqr = quartile3 - quartile1
    return iqr

def iq_range(arr):
    q1 = np.quantile(arr, 0.25)
    q3 = np.quantile(arr, 0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    
    outlier_indices = np.where((arr < lower_bound) | (arr > upper_bound))
    outlier_values = arr[outlier_indices]
    return outlier_indices, outlier_values
    

def annote(df_map, df_cl, tissue):
    depmap_id = df_cl["depmapId"]
    lineage_1 = df_cl["lineage1"]
    lineage_1_unique = list(set(lineage_1))
    # print(lineage_1_unique)
    if tissue in lineage_1_unique:
        print(f' Selecting {tissue} from the DepMap full matrix ... ')
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
        print(df_tissue)
        return df_tissue
    else:   
        msg = ' \n '.join(lineage_1_unique)
        print(f' {tissue} not present in lineage1 \n')
        print(f' Select from the following list: \n')
        print(f' {msg} \n')
        sys.exit()


def ClusterByTissues(df, df_cl, opts):
    # Function to perform Clustering on more tissues
    # tissue = opts.tissue
    msg = """ Tissue  |  No. Cell lines
--------------------------
    """
    depmap_id = df_cl["depmapId"]
    lineage_1 = df_cl["lineage1"]
    lineage_1_unique = list(set(lineage_1))
    # lineage_1_unique = [ var.replace('/', '_') for var in lineage_1_unique]
    print('\n'.join(var for var in lineage_1_unique))
    sum_shape = 0
    print(msg)
    for tissue in lineage_1_unique:
    # for tissue in  list_tissues:
        id_tissue = lineage_1[lineage_1 == tissue].index
        name_cl = depmap_id[id_tissue].tolist()
        id_true = []    
        for k, var in enumerate(df.columns):
            if var in name_cl:
                id_true.append(k)
        df_tissue = df.iloc[:, id_true]
        sum_shape += df_tissue.shape[1]
        print(f'{tissue} == {df_tissue.shape[1]}')
        if df_tissue.shape[1] != 0:
            # Remove outliers by IQR per tissue
            iqr_arr = np.empty_like(df_tissue.columns)
            for i, col in enumerate(df_tissue.columns):
                iqr_arr[i] = calculate_iqr(df_tissue, col)
            outlier_indices, outlier_values = iq_range(iqr_arr)
            print(f"outliers of {tissue}: {'  '.join(df_tissue.iloc[:,outlier_indices[0]].columns.tolist())}\n")

            df_tissue.drop(df_tissue.columns[outlier_indices[0]], axis=1, inplace=True)    # Dropping outliers
            
            # Perform clustering per tissue
            clusters_ = DoClusters(X=df_tissue, n_clusters=opts.n, mode=opts.k) 
            model = clusters_.do_clusters()
            # best_scores, best_knee = clusters_.get_score_n_knees()
            fout = f'clusters_{tissue}'
            clusters_.labels_to_csv(fout)

# def do_PCA(X, labels):
#     X = X.to_numpy()
#     pca_ = PCA(n_components=90)
#     pca_.fit(X)
#     X_pca = pca_.fit_transform(X)
#     print(f' ... Performing PCA analysis for n_component for 90% of variance')
#     print(f' explained variance ratio == {pca_.explained_variance_ratio_}')
#     count, unique = np.unique(labels)
#     fig = plt.figure()
#     for i, j in zip(count, u_labels):
#         plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1]  edgecolors='k', label= f"Cluster {i} : {j} ")


def get_csEG(file1, file2, opts):
    # Get common EG from mode of mode
    df = pd.read_csv(file1, index_col=0)
    cEG = df.index[df['0'] == 0].tolist()
    cEG_set = set(cEG)
    
    # Get context-specific EG from selected tissue
    df_t = pd.read_csv(file2, index_col=0)
    tEG = df_t.index[df_t['0'] == 1].tolist()
    tEG_set = set(tEG)
    # print('Counts from Selected Tissue:\n')
    # print(df_t.value_counts())
    fout = f'csEG_{opts.tissue}.csv'
    csEG = list(cEG_set- tEG_set)
    pd.Series(csEG).to_csv(fout, index=False)


def main():
    opts = cml_parser()
    output_dir = "output"
    f1 = os.path.join(output_dir, 'Clusters_AllTissues_DepMap.csv')
    f2 = os.path.join(output_dir, f'clusters_{opts.tissue}.csv')    
    # Get csEGs if files already exist
    if os.path.exists(f1) and os.path.exists(f2):
        get_csEG(f1, f2, opts)
    else:
        df_map, df_cl = read_input(opts)
        
        # # Drop Nan or KNN Impute data
        if opts.m == "drop":
            df_map = drop_na(df_map)
        elif opts.m == "impute":
            df_map = impute_data(df_map)
        
        # DoClusters class takes X(n_sample, n_features) DataFrame to clustering
        # full DepMap matrix
        print(f'Computing common Essential Genes (EG): full DepMap')
        st = time.time()
        clusters_all_ = DoClusters(X=df_map, n_clusters=opts.n, mode=opts.k) 
        model_all = clusters_all_.do_clusters()
        # best_scores, best_knee = clusters_.get_score_n_knees()
        clusters_all_.labels_to_csv("Clusters_AllTissues_DepMap")
        et = time.time()
        elapsed_time = et - st
        print(f" Execution time to clustering Fully DepMap :  {elapsed_time:.2f} seconds")
        
        # # Performing Clustering on specific tissues:
        print(f'Computing common Essential Genes (EG): tissue DepMap')
        st = time.time()
        ClusterByTissues(df_map, df_cl, opts)
        et = time.time()
        elapsed_time = et - st
        print(f" Execution time to clustering Fully DepMap :  {elapsed_time:.2f} seconds")
        
        # Computing context-specific EG (csEGs)
        get_csEG(f1, f2, opts)
    


if __name__ == "__main__":
    main()

