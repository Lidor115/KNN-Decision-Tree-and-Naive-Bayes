import copy
from random import randint
from sys import argv

from utils import hamming, parser, stringMaker_and_label


class knn:
    def __init__(self, train,dev, k):
        self.train = train
        self.dev = dev
        self.k = k
    def knn_a(self):
        accuracy = self.calc_accuracy(self.train,self.dev)
        return accuracy

    def calc_accuracy(self,train,dev):
        accuracy = 0
        for sample in dev:
            distace_sample = self.compute_hamming(sample[0],train)
           # distace_sample.sort(key = lambda x: x[2])
            distace_sample.sort(key = lambda x: x[0])
            yes = 0
            no = 0
            for i in range(0,self.k):
                if (distace_sample[i][1]):
                    yes+=1
                else:
                    no +=1
            label = True
            if no > yes:
                label = False
            if label == sample[1]:
                accuracy +=1
        return accuracy /len(dev)


    def compute_hamming(self,sample,train):
        distace_sample=[]
        for t in train:
            distace_sample.append((hamming(t[0], sample),t[1],t[2]))
        return distace_sample

def knn_k_folds(train_p):
    all_data, attr = stringMaker_and_label(copy.deepcopy(train_p))
    k=5
    accuracy = 0
    data = []
    one_part = int(len(all_data)/k)
    for i in range(k-1):
        data.append(all_data[i*one_part: (i+1)*one_part])
    data.append(all_data[(k-1)*one_part:])
    for i in range(k):
        dev = data[i]
        train =[]
        for j in range(k):
            if not j == i:
                train += data[j]
        acc = knn(train, dev,k).knn_a()
        accuracy +=acc
    avg_acu = "{0:.2f}".format(accuracy/k)
    print("KNN : " + str(avg_acu))
    return avg_acu

def calc_knn(train_p, dev_p,k):
    train, attr_train = stringMaker_and_label(copy.deepcopy(train_p))
    dev, attr_dev = stringMaker_and_label(copy.deepcopy(dev_p))
    acc = knn(train, dev, k).knn_a()
    avg_acu = "{0:.2f}".format(acc)
    print("total : " + str(avg_acu))
    return avg_acu




