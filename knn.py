from random import randint

from main import hamming, train


class knn:
    def __init__(self, data, k):
        self.data = data
        self.k = k
    def knn_a(self):
        folds =self.dev_train_sep()
        accuracy = 0
        for i in range(self.k):
            dev = folds[i]
            train =[]
            for j in range(self.k):
                if not j==i:
                    train+=folds[j]
            accuracy += self.calc_accuracy(train,dev)
        print("total accuracy : " + str(accuracy/self.k))
        return accuracy/self.k *100

    def dev_train_sep(self):
        div_k =[]
        all_data_number = (len(self.data))
        fold = []
        total = 0.
        one_fold =int(all_data_number/self.k)
        for i in range(self.k-1):
            fold.append(one_fold)
            total +=one_fold
        fold.append(int(all_data_number-total))
        for i in range(self.k):
            f = []
            for j in range(fold[i]):
                x = randint(0, len(self.data)-1)
                f.append(train[x])
                del(self.data[x])
            div_k.append(f)
        return div_k


    def calc_accuracy(self,train,dev):
        accuracy = 0
        for sample in dev:
            distace_sample = self.compute_hamming(sample[0],train)
            distace_sample.sort(key = lambda x: x[2])
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
        print('accuracy: ' + str(accuracy/len(dev)*100))
        return accuracy /len(dev)*100


    def compute_hamming(self,sample,train):
        distace_sample=[]
        for t in train:
            distace_sample.append((hamming(t[0], sample),t[1],t[2]))
        return distace_sample
