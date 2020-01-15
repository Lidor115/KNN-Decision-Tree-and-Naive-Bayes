

class Node:
    def __init__(self, attr,father, type):
        self.attr = attr
        self.father = father
        self.type = type
    def __str__(self):
        return self.attr
    def get_father(self):
        return self.father.attr