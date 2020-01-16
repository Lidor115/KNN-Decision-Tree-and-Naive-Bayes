import copy
from sys import argv

from main import parser, make_examples, stringMaker_and_label, max_can_eat


def dev_train_sep(k, data):
    div_k = []
    all_data_number = (len(data))
    fold = []
    total = 0.
    one_fold = int(all_data_number / k)
    for i in range(k - 1):
        fold.append(one_fold)
        total += one_fold
    fold.append(int(all_data_number - total))
    for i in range(k):
        f = data[:fold[i]]
        data = data[fold[i]:]
        div_k.append(f)
    return div_k


class NaiveByes:
    def __init__(self, train,dev, attributes,F2I):
        self.train = train
        self.F2I = F2I
        self.dev = dev
        self.attributes = attributes
        self.probs_yes = {}
        self.probs_no = {}
        self.calcProbability()
        self.prob_True = len(list(filter(lambda x: x[1] == True, train)))/ len(train)
        self.prob_False = len(list(filter(lambda x: x[1] == False, train)))/ len(train)

    def naiveBayes(self):
        acc = 0
        for d in self.dev:
            pred_good = self.prob_True
            pred_bad = self.prob_False
            for index in range(len(d[0])):
                pred_good *= self.probs_yes[(index,d[0][index])]
                pred_bad *=(self.probs_no[(index,d[0][index])])
            pred = False
            if pred_good >= pred_bad:
                pred = True
            if pred  == d[1]:
                acc +=1
        return acc/len(self.dev)*100


    def calcProbability(self):
        for attribute in self.attributes:
            index = self.F2I[attribute]
            features = set([self.train[i][0][index] for i in range(len(self.train))])
            for feature in features:


                #result = list(filter(lambda x: x[0][index] == feature, self.train))
                result_t = list(filter(lambda x: x[1]== True, self.train))
                total_t = len(result_t)
                result_f = list(filter(lambda x: x[1]== False, self.train))
                total_f= len(result_f)
                t = len(list(filter(lambda x: x[0][index] == feature, result_t)))
                f = len(list(filter(lambda x: x[0][index] == feature, result_f)))
                prob_yes= t/total_t
                prob_no = f/total_f
                self.probs_yes[(index,feature)] = prob_yes
                self.probs_no[(index,feature)] = prob_no

if __name__ == '__main__':
    train_p = parser(argv[1])
    all_ex, att = make_examples(copy.deepcopy(train_p))
    train_p_T = [[train_p[j][i] for j in range(len(train_p))] for i in range(len(train_p[0]))]
    train, F2I = stringMaker_and_label(copy.deepcopy(train_p))
    default, n = max_can_eat(train)
    k=5
    accuracy = 0
    data = dev_train_sep(k,data=all_ex)
    for i in range(k):
        dev = data[i]
        train =[]
        for j in range(k):
            if not j == i:
                train += data[j]
        naive_bayes = NaiveByes(train,dev,attributes=att,F2I=F2I)
        acc= naive_bayes.naiveBayes()
        accuracy +=acc
        print(acc)
    print("total : " + str(accuracy/k))
