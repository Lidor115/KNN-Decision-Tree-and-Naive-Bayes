from sys import argv

from ID3 import ID3_print_Tree, ID3_k_folds, write_tree
from NaiveBayes import Naive_Byse_k_folds
from knn import knn_k_folds
from utils import parser

def write_Accuracies(train_p):
    knn = knn_k_folds(train_p)
    id3= ID3_k_folds(train_p)
    naive_byse = Naive_Byse_k_folds(train_p)
    with open("accuracies.txt", 'w+') as file:
        file.write(str(id3)+"\t" +str(knn) +"\t" + str(naive_byse))
    file.close()
    return



if __name__ == '__main__':
    train_p = parser(argv[1])
    #write_Accuracies(train_p)
    tree = ID3_print_Tree(train_p)
    with open('tree.txt',"w+") as file:
        write_tree(tree,file)
    file.close()
