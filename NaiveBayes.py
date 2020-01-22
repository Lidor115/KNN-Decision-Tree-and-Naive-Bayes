import copy
from sys import argv

from utils import parser, make_examples, stringMaker_and_label, max_can_eat, dev_train_sep, parseAttributes


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
        return acc/len(self.dev)


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
def calc_Naive_Bayse(train_p,dev_p):
    train, att = make_examples(copy.deepcopy(train_p))
    dev, att_dev = make_examples(copy.deepcopy(dev_p))
    F2I = parseAttributes(train_p[0])
    naive_bayes = NaiveByes(train, dev, attributes=att, F2I=F2I)
    acc = naive_bayes.naiveBayes()
    avg_acu = "{0:.2f}".format(acc)
    print("Naive Byse : " + str(avg_acu))
    return avg_acu


def Naive_Byse_k_folds(train_p):
    all_ex, att = make_examples(copy.deepcopy(train_p))
    F2I = parseAttributes(train_p[0])
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
    avg_acu = "{0:.2f}".format(accuracy / k)
    print("Naive Byse : " + str(avg_acu))
    return avg_acu
