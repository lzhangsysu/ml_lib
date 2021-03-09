import math
import copy


"""
class definition for a tree node
label: string, store: label for leave nodes; attr for intermediate nodes
children: dictionary, indexed by attribute value
"""
class Node:
    def __init__(self, label):
        self.label = label
        self.children = dict()

    def isLeaf(self):
        return len(self.children) == 0


"""
build decision tree from data using ID3 algorithm
"""
def ID3(Data, Attributes, Labels, max_depth=10, curr_depth=0):
    ### Base Cases ###
    # all labels are the same, return a leaf
    if (len(Labels) == 1):
        label = Labels.pop()
        return Node(label)

    # attributes empty, return a leaf with most common label
    if (len(Attributes) == 0):
        label = most_common_label(Data)
        return Node(label)

    # reach max depth, return a leaf with most common label
    if curr_depth == max_depth:
        label = most_common_label(Data)
        return Node(label)

    ### Recursion ###
    # find best attribute to split to create root node
    best_attr = split_on(Data, Attributes, Labels)
    root = Node(best_attr)

    # split into subsets based on best attribute
    for attr_val in Attributes[best_attr]:
        Data_subset = get_subset(Data, best_attr, attr_val)

        # if subset is empty, add a leaf with most common label
        if len(Data_subset) == 0:
            label = most_common_label(Data)
            root.children[attr_val] = Node(label)
        else:
            # update subset attribute list
            subset_Attributes = copy.deepcopy(Attributes)
            subset_Attributes.pop(best_attr, None)

            # update subset labels set
            subset_Labels = set()
            for subset_row in Data_subset:
                subset_label = subset_row['y']
                if subset_label not in subset_Labels:
                    subset_Labels.add(subset_label)

            # recursion
            root.children[attr_val] = ID3(Data_subset, subset_Attributes, subset_Labels, max_depth, curr_depth+1)

    return root


"""
use decision tree to make prediction on a test data
return: 1 if prediction is correct, else 0
"""
def predict_hit(test_row, root):
    curr_node = root

    # tree traversal till reaching leaf
    while not curr_node.isLeaf():
        curr_attr = curr_node.label
        attr_val = test_row[curr_attr]
        curr_node = curr_node.children[attr_val]

    return 1 if curr_node.label == test_row['y'] else 0


"""
get label of test_row
"""


"""
find most common label
"""
def most_common_label(Data):
    label_counts = dict()

    for row in Data:
        label = row['y']
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    return max(label_counts.keys(), key=lambda key: label_counts[key])


"""
find the best attribute to split on
"""
def split_on(Data, Attributes, Labels):
    attr_gains = dict()

    for attribute, attr_vals in Attributes.items():
        gain = information_gain(Data, attribute, attr_vals, Labels)
        attr_gains[attribute] = gain

    return max(attr_gains.keys(), key=lambda key: attr_gains[key])


"""
calculate information gain using entropy
"""
def information_gain(Data, attribute, attr_vals, Labels):
    gain = entropy(Data, Labels)

    for val in attr_vals:
        Data_subset = get_subset(Data, attribute, val)
        gain -= (len(Data_subset) / len(Data)) * entropy(Data_subset, Labels)
    return gain


"""
get subset of data based on a certain attribute value
"""
def get_subset(Data, attribute, attr_val):
    Data_subset = []

    for row in Data:
        if row[attribute] == attr_val:
            Data_subset.append(row)

    return Data_subset


"""
calculate entropy
"""
def entropy(Data, Labels):
    props = proportions(Data, Labels).values()
    h = 0.0

    for p in props:
        if p == 0.0:
            continue
        h -= (p*math.log2(p))

    return h


"""
calculate proportions of each label
"""
def proportions(Data, Labels):
    p_dict = {label: 0 for label in Labels}

    # get label counts
    for row in Data:
        label = row['y']
        if label in p_dict:
            p_dict[label] += 1

    # convert to proportion
    for label, count in p_dict.items():
        p_dict[label] = count / len(Data) if len(Data) > 0 else 0

    return p_dict


if __name__ == '__main__':
    main()
