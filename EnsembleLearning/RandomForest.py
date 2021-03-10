import random
import ID3


def RandomForest_train(Data, Attributes, Labels, T, feature_size):
    trees = []
    m = len(Data)//10

    for t in range(0, T):
        rand_subset = [random.choice(Data) for i in range(m)]
        tree = ID3.ID3_random(rand_subset, Attributes, Labels, feature_size)
        trees.append(tree)

    return trees


def RandomForest_test(Data, trees):
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
def get_label_forest(row, trees):
    prediction = 0.0
    for tree in trees:
        label = ID3.get_label(row, tree)
        label = 1 if label == 'yes' else -1
        prediction += label
    return prediction