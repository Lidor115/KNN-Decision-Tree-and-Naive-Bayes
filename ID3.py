import copy
import math
from sys import argv

from Node import Node
from utils import make_examples, parser, stringMaker_and_label, max_can_eat, dev_train_sep, parseAttributes


class ID3:
    def __init__(self, F2I, attributes, default, examples):
        self.F2I = F2I
        self.tree = []
        self.father = None
        self.attributes = attributes
        self.best = None
        self.default = default
        self.examples = examples

    def DTL(self):
        C_False, C_True, = self.checkClassification(self.examples)
        total_examples = len(self.examples)
        if len(self.examples) == 0:
            return self.default
        elif C_True == total_examples:
            return "yes"
        elif C_False == total_examples:
            return "no"
        elif len(self.attributes) == 0:
            res = "no"
            if self.Mode(C_False, C_True) == "yes":
                res = "yes"
            return res
        else:
            self.best = self.ChooseAttribute(self.examples, C_False, C_True, self.attributes)
            features = self.attributes[self.best]
            index = self.F2I[self.best]
            for v_i in features:
                examples_i = list(filter(lambda x: x[0][index] == v_i, self.examples))
                c_f, c_t = self.checkClassification(self.examples)
                newAtt = copy.deepcopy(self.attributes)
                del newAtt[self.best]
                dtl = ID3(self.F2I, newAtt,self.Mode(c_f, c_t),examples_i)
                subtree = dtl.DTL()
                self.addNode(Node(v_i, subtree))
            return self

    def get_Features(self, examples, attribute):
        index = F2I[attribute]
        feat = set([examples[i][0][index] for i in range(len(examples))])
        return feat

    def ChooseAttribute(self, examples, C_False, C_True, attributes):
        # the first entropy
        total = C_True + C_False
        max_attribute = ""
        maxentropy = 0
        for attribute in attributes:
            entropy = self.entropy(C_False, C_True)
            features = self.attributes[attribute]
            index = self.F2I[attribute]
            for feature in features:
                c_f, c_t, total_f, total_t= self.CalcPerFeature(examples, index, feature)
                entropy -= ((c_t + c_f) / total) * self.entropy(c_f, c_t)
                #print(entropy,c_f,c_t,total)
            if entropy > maxentropy:
                maxentropy = entropy
                max_attribute = attribute
            elif entropy == maxentropy and entropy == 0.0 and max_attribute == '':
                max_attribute = attribute
                maxentropy = entropy
        return max_attribute

    def addNode(self, node):
        if not node in self.tree:
            self.tree.append(node)
        return

    def CalcPerFeature(self, examples, index, feature):
        result_True = (list(filter(lambda x: x[1] == True, examples)))
        result_False = (list(filter(lambda x: x[1] == False, examples)))
        total_f = len(result_False)
        total_t =len(result_True)
        t = len(list(filter(lambda x: x[0][index] == feature, result_True)))
        f = len(list(filter(lambda x: x[0][index] == feature, result_False)))
        return f, t, total_f, total_t

    @staticmethod
    def entropy(C_False, C_True):
        total  = C_False + C_True
        if total == 0:
            return 0
        t_ex = C_True / total
        f_ex = C_False / total
        if f_ex == 0 and t_ex == 0:
            return 0
        elif t_ex == 0:
            return -(f_ex * math.log2(f_ex))
        elif f_ex == 0:
            return -(t_ex * math.log2(t_ex))
        return -(t_ex * math.log2(t_ex) + f_ex * math.log2(f_ex))
    @staticmethod
    def checkClassification(examples):
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
                for j in range(i):
                    print('\t', end='')
                print("|", end='')
            print(str(tree.best) + "=" + subtrees[subtree_index].label, end='')
            if subtrees[subtree_index].subtree == 'yes' or subtrees[subtree_index].subtree == 'no':
                print(":" + subtrees[subtree_index].subtree, end="\n")
                if subtree_index == len(subtrees) - 1:
                    i -= 1
            else:
                i += 1
                print(end='\n')
                ID3.print_tree(subtrees[subtree_index].subtree, i)
                i -= 1

    @staticmethod
    def sort_by_label(subtree):
        return subtree.label

    @staticmethod
    def predict(tree, F2I, example, attributes,default):
        if tree == 'yes' or tree == 'no':
            return tree
        best_index = F2I[tree.best]
        example_best = example[0][best_index]
        subtree_labels = [subtree.label for subtree in tree.tree]
        subtree_index = subtree_labels.index(example_best)
        return ID3.predict(tree.tree[subtree_index].subtree, F2I, example,attributes,default)
    @staticmethod
    def get_accuracy(tree, test, F2I, attributes,default):
        accuracy = 0
        for row in test:
            label_t_f = row[1]
            label= 'no'
            if label_t_f:
                label = 'yes'
            pred = ID3.predict(tree,F2I, row, attributes,default)
            if label == pred:
                accuracy +=1
        return accuracy/len(test)

def ID3_print_Tree(train_p):
    all_ex, att = make_examples(copy.deepcopy(train_p))
    F2I = parseAttributes(train_p[0])
    default, n = max_can_eat(all_ex)
    default_yes_no = "no"
    if default:
        default_yes_no = "yes"
    d = ID3(F2I, copy.deepcopy(att),default_yes_no,copy.deepcopy(all_ex))
    tree = d.DTL()
    #d.print_tree(tree)
    return tree


def ID3_k_folds(train_p):
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
        default, n =max_can_eat(train)
        mode = "no"
        if default:
            mode = "yes"
        d = ID3(F2I, copy.deepcopy(att),mode, copy.deepcopy(train))
        tree = d.DTL()
        acc= ID3.get_accuracy(tree=copy.deepcopy(tree),test=copy.deepcopy(dev),F2I=copy.deepcopy(F2I), attributes=copy.deepcopy(att), default= mode)
        accuracy +=acc
    avg_acu = "{0:.2f}".format(accuracy / k)
    print("ID3 : " + str(avg_acu))
    return avg_acu

def write_tree_writer(tree,file):
    with open(file,"a") as f:
        write_tree(tree,f)
    f.close()

def write_tree(tree,file,i=0):
    subtrees = list(tree.tree)
    subtrees.sort(key=ID3.sort_by_label)
    for subtree_index in range(len(subtrees)):
        if i != 0:
            for j in range(i):
                file.write('\t')
            file.write("|")
        file.write("{0}={1}".format(str(tree.best),subtrees[subtree_index].label))
        #file.write(str(tree.best) +"="+  subtrees[subtree_index].label, end='')
        if subtrees[subtree_index].subtree == 'yes' or subtrees[subtree_index].subtree == 'no':
            file.write(":{0}\n".format(subtrees[subtree_index].subtree))
            if subtree_index == len(subtrees) - 1:
                i -= 1
        else:
            i += 1
            file.write('\n')
            write_tree(subtrees[subtree_index].subtree,file, i)
            i -= 1