import copy
from utils import make_examples, dev_train_sep, parseAttributes

class NaiveBayes:
    """
    class naive bayse - all the implementation of naive bayse algorithm
    get train, dev (test) the attributes of the file and F2I(feature to index)
    and classify the examples in the test-set to "yes" or "no" and get the accuracy
    """
    def __init__(self, train,dev, attributes,F2I):
        self.train = train
        self.F2I = F2I
        self.dev = dev
        self.attributes = attributes
        #probs yes and no are pre processing for all the options over the train
        self.probs_yes = {}
        self.probs_no = {}
        self.calcProbability()
        #the true and false for all of the data
        self.prob_True = len(list(filter(lambda x: x[1] == True, train)))/ len(train)
        self.prob_False = len(list(filter(lambda x: x[1] == False, train)))/ len(train)

    def naiveBayes(self):
        """
        for each example in the test-set -check the classification and
        check if right on the classification. return the total accuracy over
        all test-set
        :return: accuracy over the test-set
        """
        acc = 0
        #for each example in the test-set
        for d in self.dev:
            pred_good = self.prob_True
            pred_bad = self.prob_False
            #calc the probability for yes and no
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
        """
        calc the probability for each feature and attribute
        :return for each feature the probabily for true and the
        probability for false
        """
        for attribute in self.attributes:
            index = self.F2I[attribute]
            features = set([self.train[i][0][index] for i in range(len(self.train))])
            for feature in features:
                #all the true and false
                result_t = list(filter(lambda x: x[1]== True, self.train))
                total_t = len(result_t)
                result_f = list(filter(lambda x: x[1]== False, self.train))
                total_f= len(result_f)
                #the probability for the feature if its true or false
                t = len(list(filter(lambda x: x[0][index] == feature, result_t)))
                f = len(list(filter(lambda x: x[0][index] == feature, result_f)))
                prob_yes= t/total_t
                prob_no = f/total_f
                #assign the probabilities to the dictionaries
                self.probs_yes[(index,feature)] = prob_yes
                self.probs_no[(index,feature)] = prob_no

def calc_Naive_Bayse(train_p,dev_p):
    """
    Naive Bayse while get train and test-set
    :param train_p: train parsed
    :param dev_p: test-set parsed
    :return: the accuracy over the test-set
    """
    train, att = make_examples(copy.deepcopy(train_p))
    dev, att_dev = make_examples(copy.deepcopy(dev_p))
    F2I = parseAttributes(train_p[0])
    naive_bayes = NaiveBayes(train, dev, attributes=att, F2I=F2I)
    acc = naive_bayes.naiveBayes()
    avg_acu = "{0:.2f}".format(acc)
    return avg_acu


def Naive_Byse_k_folds(train_p):
    """
    Naive Bayse while all the data - with k-folds
    divide the data to k folds and change from test to train
    k times and return the avarage accuracy over the tests set
    :param train_p: all the data
    :return: the k-folds accuracy
    """
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
        naive_bayes = NaiveBayes(train, dev, attributes=att, F2I=F2I)
        acc= naive_bayes.naiveBayes()
        accuracy +=acc
    avg_acu = "{0:.2f}".format(accuracy / k)
    print("Naive Byse : " + str(avg_acu))
    return avg_acu
