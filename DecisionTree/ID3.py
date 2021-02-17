import math

class Node:
    def __init__(self):
        self.attribute = ""
        self.children = list()
        self.label = ""

"""
calculate proportions of each label
"""
def proportions(Data, Labels):
    p_dict = {label : 0 for label in Labels}

    # get label counts
    for row in Data:
        label = row[len(row)-1]
        if label in p_dict:
            p_dict[label] += 1

    # convert to proportion
    for label, count in p_dict.items():
        p_dict[label] = count / len(Data) if len(Data) > 0 else 0

    return p_dict


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
calculate majority error
"""
def majority_error(Data, Labels):
    props = proportions(Data, Labels).values()
    return 1 - max(props)


"""
calculate Gini index
"""
def gini_index(Data, Labels):
    props = proportions(Data, Labels).values()
    gini = 1

    for p in props:
        gini -= p**2
    
    return gini


if __name__ == '__main__':
    main()





