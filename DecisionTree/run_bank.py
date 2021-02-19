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
print("Decision Tree on Bank, a")
print("d e_trn_h e_trn_me e_trn_gi e_tst_h e_tstme e_tst_gi")

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

    print(max_depth, format(train_err_h, ".3f"), format(train_err_me, ".3f"), format(train_err_gi, ".3f"), 
    format(test_err_h, ".3f"), format(test_err_me, ".3f"), format(test_err_gi, ".3f"), sep=" & ")


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
print("Decision Tree on Bank, b")
print("d e_trn_h e_trn_me e_trn_gi e_tst_h e_tstme e_tst_gi")

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

    print(max_depth, format(train_err_h, ".3f"), format(train_err_me, ".3f"), format(train_err_gi, ".3f"), 
    format(test_err_h, ".3f"), format(test_err_me, ".3f"), format(test_err_gi, ".3f"), sep=" & ")