import copy
import math
from random import randint
from sys import argv

from Node import Node


def max_can_eat(examples):
    t = 0
    f = 0
    for example in examples:
        if example[1]:
            t += 1
        else:
            f += 1
    return t >= f, (t / (t + f))


def parser(path):
    data = []
    with open(path) as file:
        rows = file.readlines()
        for row in rows:
            data.append([feature.rstrip() for feature in row.split(sep='\t')])
    file.close()
    return data


def make_examples(train_p):
    attributes = {}
    train_p_T = [[train_p[j][i] for j in range(len(train_p))] for i in range(len(train_p[0]))]
    train_p_T.pop(-1)
    for row in train_p_T:
        attributes[row[0]] = set(row[1:])
    train_p.pop(0)
    index = 0
    all_examples = []
    for ex in train_p:
        tag = ex[-1]
        label = False
        if tag == 'yes':
            label = True
        index += 1
        all_examples.append((ex[:-1], label, index))
    return all_examples, attributes


def parseAttributes(first_line, string_label_data):
    index = 0
    all_attr = {}
    for a in first_line[:-1]:
        all_attr[a] = index
        index += 1
    return all_attr


def stringMaker_and_label(data):
    first_line = data.pop(0)
    string_label_data = []
    index = 0
    for row in data:
        label = False
        tag = row[-1]
        if tag == 'yes':
            label = True
        s = ''.join([str(x) for x in row[:-1]])
        string_label_data.append((s, label, index))
        index += 1
        attributes = parseAttributes(first_line, string_label_data)
    return string_label_data, attributes


def make_train_dev(path):
    num_lines = sum(1 for line in open(path))
    num_dev = int(num_lines * 0.2)
    dev_indexes = []
    for i in range(num_dev):
        x = randint(0, num_lines - 1)
        if not x in dev_indexes:
            dev_indexes.append(x)
    with open(path) as file:
        rows = file.readlines()
        with open('dev.txt', 'w+') as dev_file:
            with open('train.txt', 'w+') as train_file:
                dev_file.write(rows[0])
                train_file.write(rows[0])
                rows.pop(0)
                for i in range(1, num_lines - 1):
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


class DecisionTree:
    def __init__(self, F2I):
        self.F2I = F2I
        self.tree =[]
        self.father = None

    def DTL(self, examples, default, attributes):
        res_all, n = max_can_eat(examples)
        if len(examples) == 0:
            ret = "no"
            if default:
                ret = "yes"
            return (Node(ret, self.father))
        elif n == 1.0:
            return Node("yes", self.father)
        elif n == 0.0:
            return Node("no", self.father)
        elif len(attributes) == 0:
            ret = "no"
            if self.Mode(examples):
                ret = "yes"
            return (Node(ret, self.father))
        else:
            attributes_i = copy.deepcopy(attributes)
            best = self.Choose_Attribute(attributes, examples)
            best_atrr = best[0]
            node = Node(best_atrr, self.father)
            self.father= node
            features = self.get_features(examples,best_atrr)
            for feature in features:
                index = self.F2I[best_atrr]
                examples_i = list(filter(lambda x: x[0][index] == feature, examples))
                del attributes_i[best_atrr]
                self.father = Node(best_atrr + ":" + feature,self.father)
                subtree = self.DTL(examples_i,self.Mode(examples),attributes_i)
                self.tree.append(subtree)
            return self.tree
    def get_features(self,examples, attribute):
        index  = F2I[attribute]
        feat = set([examples[i][0][index] for i in range(len(examples))])
        return feat
    def Mode(self,exmples):
        t = 0
        f = 0
        for example in exmples:
            if example[1]:
                t += 1
            else:
                f += 1
        return t >= f

    def Choose_Attribute(self, attributes, examples):
        choose = []
        total_f = len(examples)
        t_f = len(list(filter(lambda x: x[1] == True, examples)))
        f_f =total_f - t_f
        for att in attributes:
            res = self.informaionGain(t_f, f_f, total_f)
            index = self.F2I[att]
            for feature in attributes[att]:
                t, f, total = self.CalcPerFeature(copy.deepcopy(examples), index, feature)
                res -= self.informaionGain(t,f,total)
            choose.append((att,res))
        choose.sort(key=lambda x: x[1],reverse=True)
        return choose[0]

    def informaionGain(self,t,f,total):
        if total == 0:
            return 0
        pos = t/total
        neg = f/total
        if pos ==0:
            return - neg*math.log(neg,2.0)
        if neg ==0:
            return -pos*math.log(pos,2.0)
        res = -pos*math.log(pos,2.0) - neg*math.log(neg,2.0)
        return res


    def CalcPerFeature(self, examples, index, feature):
        result = list(filter(lambda x: x[0][index] == feature, examples))
        total = len(result)
        t = len(list(filter(lambda x: x[1] == True, result)))
        f = total - t
        return t, f, total


# def Mode(self, examples):


if __name__ == '__main__':
    train_p = parser(argv[1])
    all_ex, att = make_examples(copy.deepcopy(train_p))
    train_p_T = [[train_p[j][i] for j in range(len(train_p))] for i in range(len(train_p[0]))]
    train, F2I = stringMaker_and_label(copy.deepcopy(train_p))
    default, n = max_can_eat(train)
    d = DecisionTree(F2I)
    d.DTL(copy.deepcopy(all_ex), default, copy.deepcopy(att))
    # knn = knn(train,5)
    # knn.knn_a()
