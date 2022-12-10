# LSH_Duplicate_Detection

Locality Sensitivity Hashing (LSH) is nowadays used as a pre-process in duplicate detection. In the main.py file, an algorithm for finding duplicate products on 4 different webshops is written. 

First, the data is cleaned such that the model words can be presented in binary vectors. These binary vectors are used to obtain a more compact signature matrix. Locality Sensitivity Hashing is then applied to to find potential duplicate pairs that will be used as
input for the clustering method. An hierarchical single linkage clustering technique is used. 

In order to evaluate the performance of the LSH, the Pair Completeness, Pair Quality and F1-measure are calculated. 


