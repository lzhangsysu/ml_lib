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


### read data ###

# read training data
with open('./bank/train.csv', 'r') as train_file:
    for line in train_file:
        example = dict()
        terms = line.strip().split(',')
        for i in range(len(terms)):
            attrName = Columns[i]
            example[attrName] = terms[i]
        
        Data_train.append(example)

train_file.close()

# read test data
with open('./bank/test.csv', 'r') as test_file:
    for line in test_file:
        example = dict()
        terms = line.strip().split(',')
        for i in range(len(terms)):
            attrName = Columns[i]
            example[attrName] = terms[i]
        
        Data_test.append(example)

test_file.close()

### convert numeric to binary using median ###

# find medians for numeric columns
medians = dict()

# find numeric columns
for attr in Columns:
    try:
        float(Data_train[0][attr])
        medians[attr] = 0.0
    except ValueError:
        pass

# calculate and store median
for attr in medians.keys():
    attr_vals = []
    for row in Data_train:
        attr_vals.append(float(row[attr]))
    medians[attr] = statistics.median(attr_vals)

# convert attribute values to binary based on median
for attr, median in medians.items():
    for row in Data_train:
        attr_val = float(row[attr])
        row[attr] = 'lo' if attr_val < median else 'hi'

    for row in Data_test:
        attr_val = float(row[attr])
        row[attr] = 'lo' if attr_val < median else 'hi'



# prediction accuracy with different max_depth
print("Decision Tree on Bank, a")
print("d e_trn_h e_trn_me e_trn_gi e_tst_h e_tstme e_tst_gi")

for max_depth in range(1, 17):
    # generate decision trees based on different purity functions
    h_tree = ID3.ID3(Data_train, Attributes, Labels, max_depth, 0)

    # count prediction hits in training set
    train_size = len(Data_train)
    train_hit_h = 0

    for row in Data_train:
        train_hit_h += ID3.predict_hit(row, h_tree)

    # calculate error rate in training set
    train_err_h = 1 - train_hit_h/train_size

    # count prediction hits in test set
    test_size = len(Data_test)
    test_hit_h = 0

    for row in Data_test:
        test_hit_h += ID3.predict_hit(row, h_tree)

    # calculate error rate in test set
    test_err_h = 1 - test_hit_h/test_size

    print(max_depth, format(train_err_h, ".3f"), format(test_err_h, ".3f"), sep=" & ")

