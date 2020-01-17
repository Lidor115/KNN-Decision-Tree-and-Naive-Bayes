import copy
import math
from sys import argv

from Node import Node
from main import parser, stringMaker_and_label, make_examples, max_can_eat


class ID3:
    def __init__(self, F2I, attributes):
        self.F2I = F2I
        self.tree = []
        self.father = None
        self.attributes = attributes
        self.best = None

    def DTL(self, examples, default, attributes):
        C_False, C_True , = self.checkClassification(examples)
        total_examples = C_True + C_False
        if len(examples) == 0:
            return default
        elif C_True == total_examples:
            return "yes"
        elif C_False == total_examples:
            return "no"
        elif len(attributes) == 0:
            res = "no"
            if self.Mode(C_False, C_True):
                res = "yes"
            return res
        else:
            self.best = self.ChooseAttribute(copy.deepcopy(examples), C_False, C_True, attributes)
            features = self.get_Features(copy.deepcopy(examples), self.best)
            index = F2I[self.best]
            for v_i in features:
                examples_i = list(filter(lambda x: x[0][index] == v_i, copy.deepcopy(examples)))
                c_f, c_t = self.checkClassification(copy.deepcopy(examples))
                newAtt = copy.deepcopy(attributes)
                if self.best in self.attributes:
                    del self.attributes[self.best]
                dtl  = ID3(self.F2I,self.attributes)
                subtree = dtl.DTL(examples_i, self.Mode(c_t, c_f), self.attributes)
                self.addNode(Node(v_i,subtree))
            return self

    def get_Features(self, examples, attribute):
        index = F2I[attribute]
        feat = set([examples[i][0][index] for i in range(len(examples))])
        return feat

    def ChooseAttribute(self, examples, C_False, C_True, attributes):
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

    def addNode(self, node):
        if not node in self.tree:
            self.tree.append(node)
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

    @staticmethod
    def print_tree(tree, i=0):
        subtrees = list(tree.tree)
        subtrees.sort(key=ID3.sort_by_label)
        for subtree_index in range(len(subtrees)):
            if i != 0:
                print("|", end='')
            print(str(tree.best) + "=" + subtrees[subtree_index].label, end='')
            if subtrees[subtree_index].subtree == 'yes' or subtrees[subtree_index].subtree == 'no':
                print(":" + subtrees[subtree_index].subtree, end="\n")
                if subtree_index == len(subtrees) - 1:
                    i -= 1
                for j in range(i):
                    print('\t', end='')
            else:
                i += 1
                print(end='\n')
                for j in range(i):
                    print('\t', end='')
                # print("|", end='')
                ID3.print_tree(subtrees[subtree_index].subtree, i)
                i -= 1

    @staticmethod
    def sort_by_label(subtree):
        return subtree.label


if __name__ == '__main__':
    train_p = parser(argv[1])
    all_ex, att = make_examples(copy.deepcopy(train_p))
    train_p_T = [[train_p[j][i] for j in range(len(train_p))] for i in range(len(train_p[0]))]
    train, F2I = stringMaker_and_label(copy.deepcopy(train_p))
    default, n = max_can_eat(train)
    d = ID3(F2I, copy.deepcopy(att))
    tree = d.DTL(copy.deepcopy(all_ex), default, copy.deepcopy(att))
    d.print_tree(tree)
    print('hi')
