from sys import argv

from ID3 import ID3_print_Tree, ID3_k_folds
from NaiveBayes import Naive_Byse_k_folds
from knn import knn_k_folds
from utils import parser

if __name__ == '__main__':
    train_p = parser(argv[1])
    knn = knn_k_folds(train_p)
    #ID3_print_Tree(train_p)
    id3= ID3_k_folds(train_p)
    naive_byse = Naive_Byse_k_folds(train_p)