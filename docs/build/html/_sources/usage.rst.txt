Get Started with EnGene
=======================

This session gives suggestions about setting up an input file for the execution of EnGene.

Input Data
##########

The input files for EnGene are file.csv (tsv or txt) containing CRISPR scores from knockout screens from project Achilles, reference selector cell lines, as well as genomic characterization data from the CCLE project. Tthese can be download from  `DepMap portal <https://depmap.org/portal/download/all/>`_ . 

The help message of python is as follow: ::

   home:$ python EnGene.py -h  
   usage: EnGene.py [-h] -i INPUT -ref REFERENCE -t TISSUE [-m {drop,impute}] [-n N] [-k {centroids,medoids,both}]
   
   optional arguments:
     -h, --help            show this help message and exit
     -m {drop,impute}      choose to drop Nan or impute
     -n N                  Number of clusters for clustering algorithms
     -k {centroids,medoids,both}
                           choose between KMeans and K-medoids clustering algorithms or both
   
   required named arguments:
     -i INPUT, --input INPUT
                           Input file CRISPR-Cas9 matrix
     -ref REFERENCE, --reference REFERENCE
                           Input reference file name
     -t TISSUE, --tissue TISSUE
                           Input tissue to be parsed; all cell lines are employed if == all


The input file is the CRISPR-Cas9 matrix downloaded from the DepMap above website, and consist of n genes on the rows and m cell lines on the columsn, while the -ref option refers to a so-called reference file which contain information about the cell lines belonging to specific tissues. This file is fundamental to screen all tissue-specific cell lines from the CRISPR scores matrix. An exmaple of **REFERENCE** file is provided in the example folder of github page.  

.. note::
    The file CRISPRGeneEffect.csv contains the knockout screens scores for all tissue cell lines.

.. warning::
    Continuous updates are added to the portal, thus, it is always suggested to employ the latest data set. 

Imputing or dropping NaN
########################

The CRISPR matrix might contain NaN which interefre with correct fucntioning of clustering algorithsm. NaN  are handled by dropping all cell lines containing NaN or by imputing them with a k-nereast-neighbour algorithm (KNN, with k = 5 ) 

K-means vs K-Medoid
###################

Two unsupervised algorithm, i.e, KMeans or KMedoid, can be chosen by the user to cluster the CRISPR matrix and assign labels to genes. While KMeans tries to minimize the within cluster sum-of-squares, KMedoids tries to minimize the sum of distances between each point and the medoid of its cluster. The medoid is a data point (unlike the centroid) which has the least total distance to the other members of its cluster.

If a binary identification is carried out, i.e. by giving -n 2,  two classes corresponding to EG and not-essential genes (NEG) are calculated.

 .. note::
     Further analysis on more clusters has not been largely tested. 

EnGene firstly performs a clustering on the full CRISPR matrix returning the common EG (cEG) and NEG to all cell lines , and the after having removed possible outliers with IQR method, it computes the EG and NEG spefic for each tissue. The output of the full matrix is reported in an file named **Clusters_AllTissues_DepMap.csv**, while for each tissue the files are names **cluster_{tissue}.csv**

 .. warning::
    Thes files must not be renamed since are used internally to compute context-specific EG  

Afterwards, the module computes the csEG for the tissue indicated by the **-t** option, subctracting from the cEG from the EG of the selected tissue.


Removal of Outliners by IQR
###########################

A further processing on the cell lines of each tissue is executed by applying the interquartile range approach to detect and remove possible outliers from the sub-matrices of each tissue. The interquartile range (IQR) is a measure of statistical dispersion, and it is defined as the difference between the 75th and 25th percentiles of the data. To calculate the IQR, the data set is divided into quartiles, or four rank-ordered even parts via linear interpolation These quartiles are denoted by Q1 (also called the lower quartile), Q2 (the median), and Q3 (also called the upper quartile). The lower quartile corresponds with the 25th percentile and the upper quartile corresponds with the 75th percentile, so IQR = Q3 −  Q1[1]. The cell lines with a IQR value out the the third and first quartile are removed from the sub-matrices.

Submission in Local 
###################

For example, the submission of a EnGene requires two input files and the tissue(s) to be investigated. For example, it can be executed in local as follows: ::

    python EnGene.py -i CRISPRGeneEffect.csv -ref cell-line-selector.csv -t Bone -m impute -n 2 -k centroids  

where the **CRISPRGeneEffect.csv** containes the CRISPR-Cas9 scores, while **cell-line-selector.csv** the cell lines corresponding to the lineages/tissues chosen. The EnGene software returns as main output a **output_bone.csv** file containing the label for each gene of the chosen tissue(s), computed by the chosen algorithm. EnGene searches for the cell lines in the CRISPRGeneEffect.csv by matching them with those from the selected tissue(s) form cell-line-selector.csv file. 

.. note::
    Currently, only one tissue can be selected. In future, multiple choices will be added.

Handling NaN
------------

EnGene handles possible NaN values in input data by performing a complete removal of all cell lines(columns) containing NaN (-m drop) or 
by executing an imputation using the `k-nearest neighbour algorithm <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_. 

Computing cEG and csEG
----------------------

The program firstly perform the clustering on the full DepMap matrix, outputting its results into the file : **Clusters_AllTissues_DepMap.csv**. Then, each tissue is processed and their results are outputted in the files name **cluster_{tissue}.csv**, which contain the cEG per tissue.  Finally, the csEG are calculated by subtracting the cEG per tissue from the cEG obtained from the DepMap full matrix.  


The possible tissues to be selected are 

.. hlist::
   :columns: 6

   * Thyroid
   * Ampulla of Vater 
   * Cervix 
   * Bone 
   * Pleura 
   * Liver 
   * Biliary Tract 
   * Bowel 
   * Normal 
   * Breast 
   * Esophagus/Stomach 
   * Unknown 
   * Uterus 
   * Fibroblast 
   * Peripheral Nervous System 
   * Other 
   * Prostate 
   * Myeloid 
   * Testis 
   * Adrenal Gland 
   * Head and Neck 
   * Ovary/Fallopian Tube 
   * Soft Tissue 
   * Lymphoid 
   * Bladder/Urinary Tract 
   * Skin 
   * Vulva/Vagina 
   * Eye 
   * Pancreas 
   * Kidney 
   * Lung 
   * CNS/Brain





Submission in Batch
###################


For instance, the submission in batch can be performed with the following template script for SLURM-managed HPC: ::

   #!/bin/bash
   # Job name:
   #SBATCH --job-name=test
   #
   # Partition:
   #SBATCH --partition=partition_name
   #
   # Request one node:
   #SBATCH --gpus-per-node=3
   #SBATCH --mem-per-gpu=1gb
   #SBATCH --output=test%j.log
   #
   #SRC=EnGene.py
   #ARG1=
   ## Command(s) to run (example):
   # python $SRC -i $ARG1 -ref $ARG2 -
   
