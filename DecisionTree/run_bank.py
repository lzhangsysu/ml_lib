import statistics
import copy
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


"""
Bank, part a
"""
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

    # print(max_depth, test_err_h, test_err_me, test_err_gi)
    # 1 0.12480000000000002 0.11660000000000004 0.11660000000000004
    # 2 0.11140000000000005 0.10880000000000001 0.10880000000000001
    # 3 0.1068 0.11180000000000001 0.11219999999999997
    # 4 0.11460000000000004 0.11619999999999997 0.122
    # 5 0.12480000000000002 0.119 0.13239999999999996
    # 6 0.136 0.12160000000000004 0.1452
    # 7 0.14439999999999997 0.12139999999999995 0.1542
    # 8 0.1452 0.12760000000000005 0.15759999999999996
    # 9 0.1522 0.13060000000000005 0.1644
    # 10 0.15480000000000005 0.13739999999999997 0.16920000000000002
    # 11 0.15659999999999996 0.14359999999999995 0.17000000000000004
    # 12 0.15800000000000003 0.15139999999999998 0.17120000000000002
    # 13 0.1582 0.15759999999999996 0.1714
    # 14 0.1582 0.1634 0.1714
    # 15 0.1582 0.16479999999999995 0.1714
    # 16 0.1582 0.16559999999999997 0.1714


"""
Bank, part b
"""
# helper function to return most common value that is not 'unknow'
def most_common_value(Data, Columns, attr):
    attr_idx = Columns.index(attr)
    val_counts = dict()

    for row in Data:
        val = row[attr_idx]
        if val != 'unknown':
            if val not in val_counts:
                val_counts[val] = 1
            else:
                val_counts[val] += 1

    return max(val_counts.keys(), key=lambda key: val_counts[key])


# data to be processed
Data_train_proc = copy.deepcopy(Data_train)
Data_test_proc = copy.deepcopy(Data_test)


# find most common values other than 'unknown'
most_common_vals = dict()

for attr, vals in Attributes.items():
    if "unknown" in vals:
        most_common_vals[attr] = most_common_value(Data_train, Columns, attr)


# replace 'unknown' with most common value
for row in Data_train_proc:
    for attr in most_common_vals:
        attr_idx = Columns.index(attr)
        if row[attr_idx] == 'unknown':
            row[attr_idx] = most_common_vals[attr]

for row in Data_test_proc:
    for attr in most_common_vals:
        attr_idx = Columns.index(attr)
        if row[attr_idx] == 'unknown':
            row[attr_idx] = most_common_vals[attr]


# prediction accuracy with different max_depth
for max_depth in range(1, 17):
    # generate decision trees based on different purity functions
    h_tree_proc = ID3.ID3(Data_train_proc, Columns, Attributes, Labels, ID3.entropy, max_depth, 0)
    me_tree_proc = ID3.ID3(Data_train_proc, Columns, Attributes, Labels, ID3.majority_error, max_depth, 0)
    gi_tree_proc = ID3.ID3(Data_train_proc, Columns, Attributes, Labels, ID3.gini_index, max_depth, 0)

    # count prediction hits in training set
    train_size = len(Data_train_proc)
    train_hit_h = 0
    train_hit_me = 0
    train_hit_gi = 0

    for row in Data_train_proc:
        train_hit_h += ID3.predict_hit(row, Columns, h_tree_proc)
        train_hit_me += ID3.predict_hit(row, Columns, me_tree_proc)
        train_hit_gi += ID3.predict_hit(row, Columns, gi_tree_proc)

    # calculate error rate in training set
    train_err_h = 1 - train_hit_h/train_size
    train_err_me = 1 - train_hit_me/train_size
    train_err_gi = 1 - train_hit_gi/train_size

    # print(max_depth, train_err_h, train_err_me, train_err_gi)
    # 1 0.11919999999999997 0.10880000000000001 0.10880000000000001
    # 2 0.10599999999999998 0.10499999999999998 0.10519999999999996
    # 3 0.10219999999999996 0.09760000000000002 0.10099999999999998
    # 4 0.08679999999999999 0.08640000000000003 0.08760000000000001
    # 5 0.07140000000000002 0.07799999999999996 0.07379999999999998
    # 6 0.05679999999999996 0.07299999999999995 0.05720000000000003
    # 7 0.04520000000000002 0.06979999999999997 0.04500000000000004
    # 8 0.03859999999999997 0.06740000000000002 0.03700000000000003
    # 9 0.03200000000000003 0.06220000000000003 0.02939999999999998
    # 10 0.02639999999999998 0.055400000000000005 0.024599999999999955
    # 11 0.023399999999999976 0.049799999999999955 0.022399999999999975
    # 12 0.022199999999999998 0.043200000000000016 0.02200000000000002
    # 13 0.02200000000000002 0.03639999999999999 0.02200000000000002
    # 14 0.02200000000000002 0.030399999999999983 0.02200000000000002
    # 15 0.02200000000000002 0.025599999999999956 0.02200000000000002
    # 16 0.02200000000000002 0.02200000000000002 0.02200000000000002

    # count prediction hits in test set
    test_size = len(Data_test_proc)
    test_hit_h = 0
    test_hit_me = 0
    test_hit_gi = 0

    for row in Data_test_proc:
        test_hit_h += ID3.predict_hit(row, Columns, h_tree_proc)
        test_hit_me += ID3.predict_hit(row, Columns, me_tree_proc)
        test_hit_gi += ID3.predict_hit(row, Columns, gi_tree_proc)

    # calculate error rate in test set
    test_err_h = 1 - test_hit_h/test_size
    test_err_me = 1 - test_hit_me/test_size
    test_err_gi = 1 - test_hit_gi/test_size

    # print(max_depth, test_err_h, test_err_me, test_err_gi)
    # 1 0.12480000000000002 0.11660000000000004 0.11660000000000004
    # 2 0.11140000000000005 0.11019999999999996 0.11040000000000005
    # 3 0.10899999999999999 0.11380000000000001 0.10819999999999996
    # 4 0.11580000000000001 0.11639999999999995 0.11519999999999997
    # 5 0.127 0.11660000000000004 0.12639999999999996
    # 6 0.13480000000000003 0.12119999999999997 0.13460000000000005
    # 7 0.14339999999999997 0.12180000000000002 0.1482
    # 8 0.14739999999999998 0.124 0.14939999999999998
    # 9 0.15500000000000003 0.12780000000000002 0.15859999999999996
    # 10 0.16359999999999997 0.13419999999999999 0.16559999999999997
    # 11 0.16080000000000005 0.14100000000000001 0.1634
    # 12 0.16180000000000005 0.14400000000000002 0.1644
    # 13 0.1614 0.15000000000000002 0.1644
    # 14 0.1614 0.15439999999999998 0.1644
    # 15 0.1614 0.15880000000000005 0.1644
    # 16 0.1614 0.15959999999999996 0.1644