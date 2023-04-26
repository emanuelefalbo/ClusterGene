Get Started with EnGene
=======================

This session gives suggestions about setting up an input file for the execution of EnGene.

Input Data
##########

The input file for EnGene is file.csv (or tsv) containing CRISPR scores from knockout screens from project Achilles, as well as genomic characterization data from the CCLE project. For eample, this can be download from  `DepMap portal <https://depmap.org/portal/download/all/>`_ . 

.. note::
    The file CRISPRGeneEffect.csv contains the knockout screens scores for all tissue cell lines.
    Therefore, context-specific cell lines must be extracted beforehand. 

.. warning::
    Continuous updates are added to the portal, thus, it is always suggested to employ the latest data set. 

Submission in Local 
###################

The submission of a EnGene in local can be executed as follows: ::

    python EnGene.py input_file.csv -m  impute -n 10 -t centroids 

The positional argument, i.e input file, is mandatory, while a series of options can be given for the analysis. The help message shows how to run the program: ::

   home:$ python EnGene.py -h  
   usage: EnGene.py [-h] [-m {drop,impute}] [-n N]
                    [-t {centroids,medoids,both}]
                    [filename]
   
   positional arguments:
     filename              input file
   
   optional arguments:
     -h, --help            show this help message and exit
     -m {drop,impute}      choose to drop Nan or impute
     -n N                  Number of clusters for clustering algorithms
     -t {centroids,medoids,both}
                           choose between KMeans and K-medoids clustering
                           algorithms or both

.. list-table:: 
   :widths: 25 50 

   * - -m
     -  choose to drop NaN or imputed data with k-nearest neighbours algorithm
   * - -n
     -  choose to the number of clusters to be tested
   * - -t
     -  choose between KMeans and K-Medoid clustering algorithms
 

The EnGene software returns as main output a **output_file.csv** containing the label for each gene, computed by the chosen algorithm.

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
   #JOB=input.csv
   ## Command(s) to run (example):
   # python $SRC $JOB
   
