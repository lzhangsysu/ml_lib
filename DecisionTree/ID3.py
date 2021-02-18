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


def ID3(Data, Columns, Attributes, Labels, puri_func, max_depth=10, curr_depth=0):
    ### Base Cases ###
    # all labels are the same, return a leaf
    if (len(Labels) == 1):
        label = Labels.pop()
        return Node(label)

    # attributes empty, return a leaf with most common label
    if (len(Attributes) == 0):
        label = most_common_label(Data, Columns)
        return Node(label)

    # reach max depth, return a leaf with most common label
    if curr_depth == max_depth:
        label = most_common_label(Data, Columns)
        return Node(label)

    ### Recursion ###
    # find best attribute to split to create root node
    best_attr = split_on(Data, Columns, Attributes, Labels, puri_func)
    root = Node(best_attr)

    # split into subsets based on best attribute
    for attr_val in Attributes[best_attr]:
        Data_subset = get_subset(Data, Columns, best_attr, attr_val)

        # if subset is empty, add a leaf with most common label
        if len(Data_subset) == 0:
            label = most_common_label(Data, Columns)
            root.children[attr_val] = Node(label)
        else:
            # update subset attribute list
            subset_Attributes = copy.deepcopy(Attributes)
            subset_Attributes.pop(best_attr, None)
            # update subset labels set
            subset_Labels = set()
            for subset_row in Data_subset:
                subset_label = subset_row[len(subset_row) - 1]
                if subset_label not in subset_Labels:
                    subset_Labels.add(subset_label)
            
            # recursion
            root.children[attr_val] = ID3(Data_subset, Columns, subset_Attributes, subset_Labels, puri_func, max_depth, curr_depth+1)
    
    return root


"""
find most common label
"""
def most_common_label(Data, Columns):
    label_idx = len(Columns) - 1
    label_counts = dict()

    for row in Data:
        label = row[label_idx]
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    return max(label_counts.keys(), key = lambda key : label_counts[key])


"""
find the best attribute to split on
"""
def split_on(Data, Columns, Attributes, Labels, puri_func):
    attr_gains = dict()

    for attribute, attr_vals in Attributes.items():
        gain = information_gain(Data, Columns, attribute, attr_vals, Labels, puri_func)
        attr_gains[attribute] = gain

    return max(attr_gains.keys(), key = lambda key : attr_gains[key])


"""
calculate information gain, puri_func: entropy, ME, gini
"""
def information_gain(Data, Columns, attribute, attr_vals, Labels, puri_func):
    gain = puri_func(Data, Labels)

    for val in attr_vals:
        Data_subset = get_subset(Data, Columns, attribute, val)
        gain -= (len(Data_subset) / len(Data)) * puri_func(Data_subset, Labels)
    return gain


"""
get subset of data based on a certain attribute value
"""
def get_subset(Data, Columns, attribute, attr_val):
    Data_subset = []

    for row in Data:
        attr_idx = Columns.index(attribute)
        if row[attr_idx] == attr_val:
            Data_subset.append(row)

    return Data_subset


"""
purity function 1: calculate entropy
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
purity function 2: calculate majority error
"""
def majority_error(Data, Labels):
    props = proportions(Data, Labels).values()
    return 1 - max(props)


"""
purity function 3: calculate Gini index
"""
def gini_index(Data, Labels):
    props = proportions(Data, Labels).values()
    gini = 1

    for p in props:
        gini -= p**2

    return gini


"""
calculate proportions of each label
"""
def proportions(Data, Labels):
    p_dict = {label: 0 for label in Labels}

    # get label counts
    for row in Data:
        label = row[len(row)-1]
        if label in p_dict:
            p_dict[label] += 1

    # convert to proportion
    for label, count in p_dict.items():
        p_dict[label] = count / len(Data) if len(Data) > 0 else 0

    return p_dict


if __name__ == '__main__':
    main()
