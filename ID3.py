import copy
import math
from sys import argv

from Node import Node
from main import parser, stringMaker_and_label, make_examples, max_can_eat, DecisionTree


class ID3:
    def __init__(self, F2I):
        self.F2I = F2I
        self.tree = []
        self.father = None


    def DTL(self, examples, default, attributes):
        C_True, C_False, = self.checkClassification(examples)
        total_examples = C_True + C_False
        if len(examples) == 0:
            return Node(default, self.father)
        elif C_True == total_examples:
            return Node("yes", self.father)
        elif C_False == total_examples:
            return Node("no", self.father)
        elif len(attributes) == 0:
            res = "no"
            if self.Mode(C_False, C_True):
                res = "yes"
            return Node(res, self.father)
        else:
            best = self.ChooseAttribute(examples, attributes, C_False, C_True)
            features = self.get_Features(examples, best)
            best_node = Node(best, self.father)
            #self.tree.append(best_node)
            self.father = best_node
            index = F2I[best]
            for v_i in features:
                examples_i = list(filter(lambda x: x[0][index] == v_i, examples))
                c_t, c_f = self.checkClassification(examples)
                if best in attributes:
                    del attributes[best]
                sub_node = Node(v_i, self.father)
                self.father = sub_node
                #self.tree.append(sub_node)
                subtree = self.DTL(examples_i, self.Mode(c_t, c_f), attributes)
                self.tree.append(subtree)
            return self.tree

    def get_Features(self, examples, attribute):
        index = F2I[attribute]
        feat = set([examples[i][0][index] for i in range(len(examples))])
        return feat

    def ChooseAttribute(self, examples, attributes, C_False, C_True):
        # the first entropy
        total = C_True + C_False
        max_attribute = ""
        maxentropy = -1
        for attribute in attributes:
            entropy = self.entropy(C_False, C_True)
            features = self.get_Features(examples, attribute)
            index = self.F2I[attribute]
            for feature in features:
                c_t, c_f = self.CalcPerFeature(examples, index, feature)
                entropy -= ((c_t + c_f) / total) * self.entropy(c_f, c_t)
            if entropy >= maxentropy:
                maxentropy = entropy
                max_attribute = attribute
        return max_attribute

    def printTree(self,subtree):
        for leaf in subtree:
            x = leaf
            s =""
            if leaf.attr == "yes" or leaf.attr == "no":
                while x:
                    s += x.attr + "=>"
                    x = x.father
                print(s)
        return

    def CalcPerFeature(self, examples, index, feature):
        result = list(filter(lambda x: x[0][index] == feature, examples))
        total = len(result)
        t = len(list(filter(lambda x: x[1] == True, result)))
        f = total - t
        return t, f

    @staticmethod
    def entropy(C_False, C_True):
        total = C_False + C_True
        if total == 0:
            return 0
        t_ex = C_True / total
        f_ex = C_False / total
        if f_ex == 0 and t_ex == 0:
            return 0
        elif t_ex == 0:
            return -(C_False * math.log2(f_ex))
        elif f_ex == 0:
            return -(t_ex * math.log2(t_ex))
        return -(t_ex * math.log2(t_ex) + f_ex * math.log2(f_ex))

    def checkClassification(self, examples):
        C_True = 0
        C_False = 0
        for example in examples:
            if example[1]:
                C_True += 1
            else:
                C_False += 1
        return C_False, C_True

    def Mode(self, C_False, C_True):
        res = 'no'
        if C_True >= C_False:
            res = 'yes'
        return res


if __name__ == '__main__':
    train_p = parser(argv[1])
    all_ex, att = make_examples(copy.deepcopy(train_p))
    train_p_T = [[train_p[j][i] for j in range(len(train_p))] for i in range(len(train_p[0]))]
    train, F2I = stringMaker_and_label(copy.deepcopy(train_p))
    default, n = max_can_eat(train)
    d = ID3(F2I)
    tree = d.DTL(copy.deepcopy(all_ex), default, copy.deepcopy(att))
    d.printTree(tree[-1])
    print('hi')
