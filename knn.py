import copy

from utils import hamming, stringMaker_and_label


class knn:
    """
    class knn - get train and dev (test) set - predict the label of an eample from the
    test set by the k elements that are most close to the example (sorted by index also)
    using hamming distance to calculate the distance
    """
    def __init__(self, train,dev, k):
        """
        initialize knn
        :param train: train-set
        :param dev: dev-set
        :param k: how many close neighbors to classify by
        """
        self.train = train
        self.dev = dev
        self.k = k
    def knn_a(self):
        """
        a method to operate the knn
        :return: accuracy over the test-set
        """
        accuracy = self.calc_accuracy(self.train,self.dev)
        return accuracy

    def calc_accuracy(self,train,dev):
        """
        calc the accuracy
        :param train: train-set
        :param dev: test-set
        :return: the accuracy over the test set
        """
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
        """
        compute hamming distance for one sample over the train-set
        :param sample: sample from test-set
        :param train: the train-set
        :return: list of distances between every example from train set to the sample
        """
        distace_sample=[]
        for t in train:
            distace_sample.append((hamming(t[0], sample),t[1],t[2]))
        return distace_sample

def knn_k_folds(train_p):
    """
    k- folds for KNN, get data-set and divide to k (k=5)
    return the avarage accuracy over all the the k times we runs the algorithm
    each time one fold is the test-set and the other folds are train set
    :param train_p: all dataset
    :return:
    """
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
    return avg_acu

def calc_knn(train_p, dev_p,k):
    """
    calculate knn by train,test and k
    :param train_p: train-set
    :param dev_p: test-set
    :param k: how many close neighbors to classify by
    :return: accuracy over the test-set
    """
    train, attr_train = stringMaker_and_label(copy.deepcopy(train_p))
    dev, attr_dev = stringMaker_and_label(copy.deepcopy(dev_p))
    acc = knn(train, dev, k).knn_a()
    avg_acu = "{0:.2f}".format(acc)
    return avg_acu




