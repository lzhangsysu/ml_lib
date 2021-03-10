import ID3
import random


"""
Train data with T iterations using bagging, return each of T trees
"""
def Bagging_train(Data, Attributes, Labels, T):
    trees = []
    m = len(Data)//10

    for t in range(0, T):
        # draw random subset with replacement and run ID3 on subset
        rand_subset = [random.choice(Data) for i in range(m)]
        tree = ID3.ID3_weighted(rand_subset, Attributes, Labels, None, 0)
        trees.append(tree)

    return trees


"""
Calculate prediction hit rate
"""
def Bagging_test(Data, trees):
    hit = 0

    for row in Data:
        prediction = 0.0
        for tree in trees:
            label = ID3.get_label(row, tree)
            label = 1 if label == 'yes' else -1
            prediction += label

        if row['y'] == 'yes' and prediction > 0:
            hit += 1
        if row['y'] == 'no' and prediction < 0:
            hit += 1
    
    return hit/float(len(Data))


"""
find predicted label for a row
"""
def get_label_bagging(row, trees):
    prediction = 0.0
    for tree in trees:
        label = ID3.get_label(row, tree)
        label = 1 if label == 'yes' else -1
        prediction += label
    return prediction
