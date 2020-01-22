

class Node:
    """
    node class - label (of the node yes or no) and the subtree
    this class is for the ID3 algorithm that add nodes to the tree
    """
    def __init__(self, label,subtree):
        """
        initialize the node
        :param label: the label of the node (yes or no)
        :param subtree: the subtree
        """
        self.label = label
        self.subtree = subtree

    def __str__(self):
        """
        return the label of the node
        :return: the label of the node
        """
        return self.label
