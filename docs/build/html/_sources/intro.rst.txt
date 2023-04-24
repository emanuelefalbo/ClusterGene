Introduction to EnGene
======================

This is a manual for the EnGene software which is currently employed to analyse omics data, i.e. gene scores,  from distributed database

Theoretical Background 
######################

When analysing data from CRISPR-Cas9 screens in functional and translational studies another major computational problem is to classify and distinguish genetic dependencies involved in normal essential biological processes from disease- and genomic-context-specific vulnerabilities. Identifying context-specific essential genes, and distinguishing them from constitutively essential genes shared across all tissues and cells, i.e. core-fitness genes (CFGs), is also crucial for elucidating the mechanisms involved in tissue-specific diseases. In this prospective, focusing on very well-defined genomic contexts in tumours allows identifying cancer synthetic lethalities that could be exploited therapeutically.

Gene dependency profiles, generated via pooled CRISPR-Cas9 screening across large panels of human cancer cell lines, are becoming increasingly available. However, identifying and discriminating CFGs and context-specific essential genes from this type of functional genetics screens remains not trivial.
To this end, we present the EnGene software that aims to the solve the essential gene(EG) classification by performing the following four tasks:

#. Data structuring: assembly data from different sources
#. Gene attributes: add biological attributes to the data in step 1.
#. Class labeling: Determine labels of EG according both supervised and unspervised learning approaches. 
#. Model classification: Train a supervised algorithm for EG classification
