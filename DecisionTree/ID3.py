import math

"""
class definition for a tree node
"""
class Node:
    def __init__(self):
        self.attribute = ""
        self.children = list()

"""
find most common value on an attribute
"""
def most_common_val(Data, Columns, attribute):
    attr_idx = Columns.index(attribute)
    val_counts = dict()

    for row in Data:
        val = row[attr_idx]
        if val not in val_counts:
            val_counts[val] = 1
        else:
            val_counts[val] += 1

    return max(val_counts.keys(), key = lambda key : val_counts[key])


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
def get_subset(Data, Columns, attribute, val):
    Data_subset = []

    for row in Data:
        attr_idx = Columns.index(attribute)
        if row[attr_idx] == val:
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
