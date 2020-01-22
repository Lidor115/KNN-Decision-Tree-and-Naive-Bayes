import copy
import math
from Node import Node
from utils import make_examples, max_can_eat, dev_train_sep, parseAttributes


class ID3:
    """
    Class ID3 - get attributes (all attributes and the features of the train)
    the default (yes or no) and the examples(train-set)
    build the tree as a Decision-tree implementation of ID3
    """
    def __init__(self, F2I, attributes, default, examples):
        self.F2I = F2I
        self.tree = []
        self.attributes = attributes
        self.best = None
        self.default = default
        self.examples = examples

    def DTL(self):
        """
        main function of the algorithm ID3.
        works recursive and after all return a full tree of ID3
        :return: tree
        """
        C_False, C_True, = self.checkClassification(self.examples)
        total_examples = len(self.examples)
        #if there are no examples
        if len(self.examples) == 0:
            return self.default
        #all examples are yes
        elif C_True == total_examples:
            return "yes"
        #all examples are no
        elif C_False == total_examples:
            return "no"
        #there are no more attributes
        elif len(self.attributes) == 0:
            res = "no"
            if self.Mode(C_False, C_True) == "yes":
                res = "yes"
            return res
        else:
            #choose the best attribute by entropy
            self.best = self.ChooseAttribute(self.examples, C_False, C_True, self.attributes)
            #get all the features
            features = self.attributes[self.best]
            index = self.F2I[self.best]
            #for every feature in the attribute
            for v_i in features:
                examples_i = list(filter(lambda x: x[0][index] == v_i, self.examples))
                #c_f and c_t is sum of "no" classification and sum of "yes" classifications
                c_f, c_t = self.checkClassification(self.examples)
                #remove the best attribute from attributes
                newAtt = copy.deepcopy(self.attributes)
                del newAtt[self.best]
                #recursive the new examples with new attributes(without best)
                dtl = ID3(self.F2I, newAtt,self.Mode(c_f, c_t),examples_i)
                #call to subtree (will be yes or no at the end)
                subtree = dtl.DTL()
                #add the node to the tree
                self.addNode(Node(v_i, subtree))
            return self

    def get_Features(self, examples, attribute):
        """
        get only relevent features for the attribute
        :param examples: the relevent examples
        :param attribute: attribute
        :return:only relevent features for the attribute
        """
        index = self.F2I[attribute]
        feat = set([examples[i][0][index] for i in range(len(examples))])
        return feat

    def ChooseAttribute(self, examples, C_False, C_True, attributes):
        """
        choose the best attribute by entropy (and calculate the gain)
        :param examples: all the relevent examples
        :param C_False: how many "no" examples
        :param C_True: how many "yes" examples
        :param attributes: all the attributes
        :return: best attribute
        """
        # the first entropy
        total = C_True + C_False
        max_attribute = ""
        maxentropy = 0
        #check foreach attribute
        for attribute in attributes:
            #calc the first entropy over the data
            entropy = self.entropy(C_False, C_True)
            #get all features for the attribute
            features = self.attributes[attribute]
            #get the index of the attribute
            index = self.F2I[attribute]
            for feature in features:
                #calc per feature how many yes and no examples
                c_f, c_t, total_f, total_t= self.CalcPerFeature(examples, index, feature)
                #calc the entropy
                entropy -= ((c_t + c_f) / total) * self.entropy(c_f, c_t)
                #print(entropy,c_f,c_t,total)
            #change the max entropy and "best" attribute
            if entropy > maxentropy:
                maxentropy = entropy
                max_attribute = attribute
            elif entropy == maxentropy and entropy == 0.0 and max_attribute == '':
                max_attribute = attribute
                maxentropy = entropy
        return max_attribute

    def addNode(self, node):
        """
        add the node to the tree
        :param node: node
        :return: nothing
        """
        if not node in self.tree:
            self.tree.append(node)
        return

    def CalcPerFeature(self, examples, index, feature):
        """
        calculate how many yes and no examples there are per feature
        :param examples: all relevent examples
        :param index: the attribute index
        :param feature: the feature
        :return: how many: no examples,yes examples,total no examples,total yes examples
        """
        result_True = (list(filter(lambda x: x[1] == True, examples)))
        result_False = (list(filter(lambda x: x[1] == False, examples)))
        total_f = len(result_False)
        total_t =len(result_True)
        t = len(list(filter(lambda x: x[0][index] == feature, result_True)))
        f = len(list(filter(lambda x: x[0][index] == feature, result_False)))
        return f, t, total_f, total_t

    @staticmethod
    def entropy(C_False, C_True):
        """
        calculate the entropy
        :param C_False: how many no examples
        :param C_True: how many yes examples
        :return: the entropy
        """
        total  = C_False + C_True
        #there are no examples
        if total == 0:
            return 0
        t_ex = C_True / total
        f_ex = C_False / total
        if f_ex == 0 and t_ex == 0:
            return 0
        #only no examples
        elif t_ex == 0:
            return -(f_ex * math.log2(f_ex))
        #only yes examples
        elif f_ex == 0:
            return -(t_ex * math.log2(t_ex))
        return -(t_ex * math.log2(t_ex) + f_ex * math.log2(f_ex))
    @staticmethod
    def checkClassification(examples):
        """
        check how many yes and no examples
        :param examples: all relevent examples
        :return: how many yes and no examples
        """
        C_True = 0
        C_False = 0
        for example in examples:
            if example[1]:
                C_True += 1
            else:
                C_False += 1
        return C_False, C_True

    def Mode(self, C_False, C_True):
        """
        :param C_False: how many no examples
        :param C_True: how many yes examples
        :return: yes or no
        """
        res = 'no'
        if C_True >= C_False:
            res = 'yes'
        return res

    @staticmethod
    def print_tree(tree, i=0):
        """
        print tree to screen (not necessary for this assignment)
        :param tree: the tree we want to print
        :param i: the index of the tree
        :return: nothing
        """
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
        """
        sort the tree by label
        :param subtree: the tree
        :return: sort of the tree
        """
        return subtree.label

    @staticmethod
    def predict(tree, F2I, example, attributes,default):
        """
        predict for "yes" or "no" for the example
        :param tree: the tree
        :param F2I: feature to index dictionary
        :param example: the example we want to check
        :param attributes: a dictionary for the fearure and the attributes
        :param default: the default answer (if there is no attribute in F2I)
        :return: the prediction for the example
        """
        if tree == 'yes' or tree == 'no':
            return tree
        best_index = F2I[tree.best]
        example_best = example[0][best_index]
        subtree_labels = [subtree.label for subtree in tree.tree]
        subtree_index = subtree_labels.index(example_best)
        return ID3.predict(tree.tree[subtree_index].subtree, F2I, example,attributes,default)

    @staticmethod
    def get_accuracy(tree, test, F2I, attributes,default):
        """
        :param tree: the tree
        :param test: the test-ser
        :param F2I: feature to index dictionary
        :param attributes: a dictionary for the fearure and the attributes
        :param default: the default answer (if there is no attribute in F2I)
        :return: the accuracy over the test-set
        """
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
    """
    print the tree
    :param train_p:the train-set
    :return: print the tree
    """
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
    """
    k- folds for ID3, get data-set and divide to k (k=5)
    return the avarage accuracy over all the the k times we runs the algorithm
    each time one fold is the test-set and the other folds are train set
    :param train_p: all the data
    :return: avarage accuracy over all the the k times
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
        default, n =max_can_eat(train)
        mode = "no"
        if default:
            mode = "yes"
        d = ID3(F2I, copy.deepcopy(att),mode, copy.deepcopy(train))
        tree = d.DTL()
        acc= ID3.get_accuracy(tree=copy.deepcopy(tree),test=copy.deepcopy(dev),F2I=copy.deepcopy(F2I), attributes=copy.deepcopy(att), default= mode)
        accuracy +=acc
    avg_acu = "{0:.2f}".format(accuracy / k)
    return avg_acu,tree

def calc_ID3(train_p,dev_p):
    """
    :param train_p: train-set
    :param dev_p: dev-set
    :return:the accuracy (and the tree)
    """
    train, att = make_examples(copy.deepcopy(train_p))
    dev,att_dev = make_examples(copy.deepcopy(dev_p))
    F2I = parseAttributes(train_p[0])
    default, n = max_can_eat(train)
    mode = "no"
    if default:
        mode = "yes"
    d = ID3(F2I, copy.deepcopy(att),mode, copy.deepcopy(train))
    tree = d.DTL()
    acc= ID3.get_accuracy(tree=copy.deepcopy(tree),test=copy.deepcopy(dev),F2I=copy.deepcopy(F2I), attributes=copy.deepcopy(att), default= mode)
    avg_acu = "{0:.2f}".format(acc)
    return avg_acu,tree

def write_tree(tree,file,i=0):
    """
    write the tree for file as a tree
    :param tree: the tree
    :param file: the file (have to be open and can write in)
    :param i: the iteration (over the tree)
    :return: nothing - write into the file
    """
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