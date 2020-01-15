

class Node:
    def __init__(self, attr,father):
        self.attr = attr
        self.father = father
    def __str__(self):
        return self.attr
    def get_father(self):
        return self.father.attr