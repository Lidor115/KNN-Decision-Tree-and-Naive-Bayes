from random import randint
from sys import argv


def parser(path):
    data = []
    with open(path) as file:
        rows = file.readlines()
        for row in rows:
            data.append([feature.rstrip() for feature in row.split(sep='\t')])
    file.close()
    return data

def stringMaker_and_label(data):
    first_line = data.pop(0)
    string_label_data = []
    for row in data:
        label = False
        tag = row[-1]
        if tag == 'yes':
            label = True
        s =''.join([str(x) for x in row[:-1]])
        string_label_data.append((s,label))
    return string_label_data

def make_train_dev(path):
    num_lines = sum(1 for line in open(path))
    num_dev = int(num_lines*0.2)
    dev_indexes =[]
    for i in range(num_dev):
        x =randint(0,num_lines-1)
        if not x in dev_indexes:
            dev_indexes.append(x)
    with open(path) as file:
        rows = file.readlines()
        with open('dev.txt','w+') as dev_file:
            with open('train.txt', 'w+') as train_file:
                dev_file.write(rows[0])
                train_file.write(rows[0])
                rows.pop(0)
                for i in range(1,num_lines-1):
                    if i in dev_indexes:
                        dev_file.write(rows[i])
                    else:
                        train_file.write(rows[i])
            train_file.close()
        dev_file.close()
    file.close()

def hamming(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length.")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

class knn:
    def __init__(self, train, k, dev):
        self.train = train
        self.k = k
        self.dev = dev
    def knn_a(self):
        accuracy = 0
        for sample in self.dev:
            distace_sample = self.compute_hamming(sample[0])
            distace_sample.sort()
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
        print('accuracy: ' + str(accuracy/len(self.dev)*100))



    def compute_hamming(self,sample):
        distace_sample=[]
        for t in self.train:
            distace_sample.append((hamming(t[0], sample),t[1]))
        return distace_sample



if __name__ == '__main__':
    make_train_dev('./dataset.txt')
    train_p = parser(argv[1])
    dev_p = parser(argv[2])
    train = stringMaker_and_label(train_p)
    dev = stringMaker_and_label(dev_p)
    knn = knn(train,5,dev)
    knn.knn_a()


