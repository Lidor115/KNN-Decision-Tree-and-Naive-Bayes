import copy
from sys import argv

from ID3 import ID3_print_Tree, ID3_k_folds, write_tree, calc_ID3
from NaiveBayes import Naive_Byse_k_folds, calc_Naive_Bayse
from knn import knn_k_folds, calc_knn
from utils import parser, make_train_dev


def write_Accuracies(train_p,file):
    knn = knn_k_folds(train_p)
    id3,tree= ID3_k_folds(train_p)
    naive_byse = Naive_Byse_k_folds(train_p)
    file.write(str(id3)+"\t" +str(knn) +"\t" + str(naive_byse))
    return

def write_original_dataset():
    train_p = parser(argv[1])
    tree = ID3_print_Tree(copy.deepcopy(train_p))
    with open("accuracies.txt", 'w+') as file_acc:
         write_Accuracies(copy.deepcopy(train_p), file_acc)
    file_acc.close()
    with open('tree.txt',"w+") as file:
        write_tree(tree,file)
    file.close()

def write_and_calc():
    train_p = parser("train.txt")
    dev_p = parser("test.txt")
    knn = calc_knn(train_p,dev_p,5)
    id3, tree =calc_ID3(train_p,dev_p)
    naive_bayse =calc_Naive_Bayse(train_p,dev_p)
    with open('output.txt',"w+") as file:
        write_tree(tree,file)
        file.write(str(id3) + "\t" + str(knn) + "\t" + str(naive_bayse))
    file.close()


if __name__ == '__main__':
    write_and_calc()