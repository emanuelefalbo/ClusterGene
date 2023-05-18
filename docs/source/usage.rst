Get Started with EnGene
=======================

This session gives suggestions about setting up an input file for the execution of EnGene.

Input Data
##########

The input files for EnGene are file.csv (tsv or txt) containing CRISPR scores from knockout screens from project Achilles, reference selector cell lines, as well as genomic characterization data from the CCLE project. Tthese can be download from  `DepMap portal <https://depmap.org/portal/download/all/>`_ . 

.. note::
    The file CRISPRGeneEffect.csv contains the knockout screens scores for all tissue cell lines.
    Therefore, context-specific cell lines must be extracted beforehand. 

.. warning::
    Continuous updates are added to the portal, thus, it is always suggested to employ the latest data set. 

Submission in Local 
###################

For example, the submission of a EnGene requires two input files and the tissue(s) to be investigated. For example, it can be executed in local as follows: ::

    python EnGene.py -i CRISPRGeneEffect.csv -ref cell-line-selector.csv -t Bone -m impute -n 2 -k centroids -o output_bone 

where the **CRISPRGeneEffect.csv** containes the CRISPR-Cas9 scores, while **cell-line-selector.csv** the cell lines corresponding to the lineages/tissues chosen. The EnGene software returns as main output a **output_bone.csv** file containing the label for each gene of the chosen tissue(s), computed by the chosen algorithm. EnGene searches for the cell lines in the CRISPRGeneEffect.csv by matching them with those from the selected tissue(s) form cell-line-selector.csv file. 

.. note::
    Currently, only one or all tissues can be selected, if **all** string is given to **-t** all cell lines are employed. 
    In future, multiple choices will be added.

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


The help message of python is self-explanatory: ::

   home:$ python EnGene.py -h  
   usage: EnGene.py [-h] -i INPUT -ref REFERENCE -t TISSUE [-o OUTPUT] [-m {drop,impute}] [-n N] [-k {centroids,medoids,both}]
   
   optional arguments:
     -h, --help            show this help message and exit
     -o OUTPUT, --output OUTPUT
                           name output file
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
   
