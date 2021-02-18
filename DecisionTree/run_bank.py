import statistics

import ID3

Data_train = []
Data_test = []

Columns = [
    'age', # numeric
    'job',
    'marital',
    'education',
    'default',
    'balance', # numeric
    'housing',
    'loan',
    'contact',
    'day', # numeric
    'month',
    'duration', # numeric
    'campaign', # numeric
    'pdays', # numeric
    'previous', # numeric
    'poutcome',
    'y'
]

Attributes = {
    'age': ['lo', 'hi'],
    'job': [
        "admin.","unknown","unemployed","management",
        "housemaid","entrepreneur","student","blue-collar",
        "self-employed","retired","technician","services"
        ],
    'marital': ["married","divorced","single"],
    'education': ["unknown","secondary","primary","tertiary"],
    'default': ["yes","no"],
    'balance':['lo', 'hi'],
    'housing': ["yes","no"],
    'loan': ["yes","no"],
    'contact': ["unknown","telephone","cellular"],
    'day': ['lo', 'hi'],
    'month': ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
    'duration': ['lo', 'hi'],
    'campaign': ['lo', 'hi'],
    'pdays': ['lo', 'hi'],
    'previous': ['lo', 'hi'],
    'poutcome': ["unknown","other","failure","success"],
}

Labels = {"yes","no"}


# read training data
with open('./bank/train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        Data_train.append(terms)

train_file.close()

# read test data
with open('./bank/test.csv', 'r') as test_file:
    for line in test_file:
        terms = line.strip().split(',')
        Data_test.append(terms)

test_file.close()


# find medians for numeric columns
medians = dict()

# find numeric columns
for attr in Columns:
    try:
        float(Data_train[0][Columns.index(attr)])
        medians[attr] = 0.0
    except ValueError:
        pass

# calculate and store median
for attr in medians.keys():
    attr_vals = []
    for row in Data_train:
        attr_vals.append(float(row[Columns.index(attr)]))
    medians[attr] = statistics.median(attr_vals)


# convert attribute values to binary based on median
for attr, median in medians.items():
    for row in Data_train:
        attr_val = float(row[Columns.index(attr)])
        row[Columns.index(attr)] = 'lo' if attr_val < median else 'hi'

    for row in Data_test:
        attr_val = float(row[Columns.index(attr)])
        row[Columns.index(attr)] = 'lo' if attr_val < median else 'hi'


# prediction accuracy with different max_depth
for max_depth in range(1, 17):
    # generate decision trees based on different purity functions
    h_tree = ID3.ID3(Data_train, Columns, Attributes, Labels, ID3.entropy, max_depth, 0)
    me_tree = ID3.ID3(Data_train, Columns, Attributes, Labels, ID3.majority_error, max_depth, 0)
    gi_tree = ID3.ID3(Data_train, Columns, Attributes, Labels, ID3.gini_index, max_depth, 0)

    # count prediction hits in training set
    train_size = len(Data_train)
    train_hit_h = 0
    train_hit_me = 0
    train_hit_gi = 0

    for row in Data_train:
        train_hit_h += ID3.predict_hit(row, Columns, h_tree)
        train_hit_me += ID3.predict_hit(row, Columns, me_tree)
        train_hit_gi += ID3.predict_hit(row, Columns, gi_tree)

    # calculate error rate in training set
    train_err_h = 1 - train_hit_h/train_size
    train_err_me = 1 - train_hit_me/train_size
    train_err_gi = 1 - train_hit_gi/train_size

    # print(max_depth, train_err_h, train_err_me, train_err_gi)
    # 1 0.11919999999999997 0.10880000000000001 0.10880000000000001
    # 2 0.10599999999999998 0.10419999999999996 0.10419999999999996
    # 3 0.10060000000000002 0.09599999999999997 0.09340000000000004
    # 4 0.07920000000000005 0.0826 0.07479999999999998
    # 5 0.06120000000000003 0.0706 0.059599999999999986
    # 6 0.04720000000000002 0.06520000000000004 0.04679999999999995
    # 7 0.03480000000000005 0.06240000000000001 0.034599999999999964
    # 8 0.02859999999999996 0.05600000000000005 0.026599999999999957
    # 9 0.02300000000000002 0.04859999999999998 0.021199999999999997
    # 10 0.017000000000000015 0.04239999999999999 0.017000000000000015
    # 11 0.014399999999999968 0.03720000000000001 0.014599999999999946
    # 12 0.013599999999999945 0.02959999999999996 0.013800000000000034
    # 13 0.013599999999999945 0.025000000000000022 0.013599999999999945
    # 14 0.013599999999999945 0.01980000000000004 0.013599999999999945
    # 15 0.013599999999999945 0.015599999999999947 0.013599999999999945
    # 16 0.013599999999999945 0.013599999999999945 0.013599999999999945

    # count prediction hits in test set
    test_size = len(Data_test)
    test_hit_h = 0
    test_hit_me = 0
    test_hit_gi = 0

    for row in Data_test:
        test_hit_h += ID3.predict_hit(row, Columns, h_tree)
        test_hit_me += ID3.predict_hit(row, Columns, me_tree)
        test_hit_gi += ID3.predict_hit(row, Columns, gi_tree)

    # calculate error rate in test set
    test_err_h = 1 - test_hit_h/test_size
    test_err_me = 1 - test_hit_me/test_size
    test_err_gi = 1 - test_hit_gi/test_size

    print(max_depth, test_err_h, test_err_me, test_err_gi)