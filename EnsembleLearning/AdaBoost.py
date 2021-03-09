import ID3
import math

def AdaBoost_Train(Data, Attributes, Labels, T):
    trees = []
    alphas = []

    for t in range(0, T):
        # train a 2-layer tree
        tree = ID3.ID3_weighted(Data, Attributes, Labels, 1, 0)
        trees.append(tree)

        # calculate votes
        err = ID3.weighted_err(Data, tree)
        # err_test = ID3.weighted_err(Data_test, tree)
        # print(err, err_test)
        alpha = 0.5 * math.log((1-err)/err)
        alphas.append(alpha)

        # update weights
        norm = 0.0
        for row in Data:
            label = ID3.get_label(row, tree)
            # calculate weight depending on whether label is correct
            if label != row['y']:
                newWeight = row['weight'] * math.exp(alpha)
            else:
                newWeight = row['weight'] * math.exp(-alpha)

            row['weight'] = newWeight
            norm += newWeight

        # normalize weights
        for row in Data:
            row['weight'] /= norm

    return trees, alphas


def AdaBoost_Test(Data, trees, alphas):
    hit = 0

    for row in Data:
        prediction = 0.0
        for tree, alpha in zip(trees, alphas):
            label = ID3.get_label(row, tree)
            label = 1 if label == 'yes' else -1
            prediction += label * alpha

        if row['y'] == 'yes' and prediction > 0:
            hit += 1
        if row['y'] == 'no' and prediction < 0:
            hit += 1
    
    return hit/float(len(Data))

def print_err_Ada(Data_train, Data_test, Attributes, Labels, T):
    errors_train = []
    errors_test = []

    for t in range(0, T):
        # train a 2-layer tree
        tree = ID3.ID3_weighted(Data_train, Attributes, Labels, 1, 0)

        # calculate votes
        err_train = ID3.weighted_err(Data_train, tree)
        err_test = ID3.weighted_err(Data_test, tree)
        errors_train.append(err_train)
        errors_test.append(err_test)
        alpha = 0.5 * math.log((1-err_train)/err_train)

        # update weights
        norm = 0.0
        for row in Data_train:
            label = ID3.get_label(row, tree)
            # calculate weight depending on whether label is correct
            if label != row['y']:
                newWeight = row['weight'] * math.exp(alpha)
            else:
                newWeight = row['weight'] * math.exp(-alpha)

            row['weight'] = newWeight
            norm += newWeight

        # normalize weights
        for row in Data_train:
            row['weight'] /= norm

    return errors_train, errors_test

