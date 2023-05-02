# EnGene
software for analysis of gene data sets

usage: EnGene.py [-h] -i INPUT -ref REFERENCE -t TISSUE [-o OUTPUT] [-m {drop,impute}] [-n N] [-k {centroids,medoids,both}]

options:
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
                        Input tissue to be parsed

