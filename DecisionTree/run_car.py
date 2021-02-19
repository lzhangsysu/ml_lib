import ID3

Data_train = []
Data_test = []

Columns = [
    'buying',
    'maint',
    'doors',
    'persons',
    'lug_boot',
    'safety',
    'label'
]

Attributes = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

Labels = {'unacc', 'acc', 'good', 'vgood'}


# read training data
with open('./car/train.csv', 'r') as train_file:
    for line in train_file:
        row = line.strip().split(',')
        Data_train.append(row)

train_file.close()

# read test data
with open('./car/test.csv', 'r') as test_file:
    for line in test_file:
        row = line.strip().split(',')
        Data_test.append(row)

test_file.close()

print("Decision Tree on Car")
print("d e_trn_h e_trn_me e_trn_gi e_tst_h e_tstme e_tst_gi")

# prediction accuracy with different max_depth
for max_depth in range(1, 7):
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


