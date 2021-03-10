import statistics
import copy
import ID3
import RandomForest
import matplotlib.pyplot as plt
import random
import numpy as np

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

### assign uniform weight to each data ###
for row in Data_train:
    row['weight'] = 1/float(len(Data_train))

for row in Data_test:
    row['weight'] = 1/float(len(Data_test))


def reset_weight(Data):
    w0 = 1/float(len(Data))
    for row in Data:
        row['weight'] = w0

# Q2.d
# Run random forest for T = 1-500, with feature number 2,4,6, plot error-T relationship
outFile = open("forest_out.txt", 'w')
outFile.write("iter\terr_train\terr_test\tN=2\n")

for T in [1,2,3,4,5,6,8,10,15,20,30,50,70,90,120,150,200,250,300,350,400,450,500]:
    trees = RandomForest.RandomForest_train(Data_train, Attributes, Labels, T, 2)
    hit_train = RandomForest.RandomForest_test(Data_train, trees)
    hit_test = RandomForest.RandomForest_test(Data_test, trees)
    outFile.write(str(T) + "\t" + str(1-hit_train) + "\t" + str(1-hit_test) + "\n")
    reset_weight(Data_train)

outFile.write('\n\n')
outFile.write("iter\terr_train\terr_test\tN=4\n")
for T in [1,2,3,4,5,6,8,10,15,20,30,50,70,90,120,150,200,250,300,350,400,450,500]:
    trees = RandomForest.RandomForest_train(Data_train, Attributes, Labels, T, 4)
    hit_train = RandomForest.RandomForest_test(Data_train, trees)
    hit_test = RandomForest.RandomForest_test(Data_test, trees)
    outFile.write(str(T) + "\t" + str(1-hit_train) + "\t" + str(1-hit_test) + "\n")
    reset_weight(Data_train)

outFile.write('\n\n')
outFile.write("iter\terr_train\terr_test\tN=6\n")
for T in [1,2,3,4,5,6,8,10,15,20,30,50,70,90,120,150,200,250,300,350,400,450,500]:
    trees = RandomForest.RandomForest_train(Data_train, Attributes, Labels, T, 6)
    hit_train = RandomForest.RandomForest_test(Data_train, trees)
    hit_test = RandomForest.RandomForest_test(Data_test, trees)
    outFile.write(str(T) + "\t" + str(1-hit_train) + "\t" + str(1-hit_test) + "\n")
    reset_weight(Data_train)

outFile.close()


# Q2.e
# draw 100 random sample and run bagging with T=500 for each sample
def sample_no_replacement(Data, size):
    Data_copy = copy.deepcopy(Data)
    sample = []
    for i in range(size):
        rand_idx = random.randint(0, len(Data_copy)-1)
        sample.append(Data_copy[rand_idx])
        del Data_copy[rand_idx]
    return sample

# obtain set of trees
predictors = []
for i in range(100):
    sample = sample_no_replacement(Data_train, 1000)
    trees = RandomForest.RandomForest_train(sample, Attributes, Labels, 500, 4)
    predictors.append(trees)

# calculate single tree bias and variance
sum_single_bias = 0.0
sum_single_var = 0.0
for row in Data_test:
    avg = 0
    row_predictions = []
    # find individual predictions and mean prediction
    for trees in predictors:
        tree = trees[0]
        label = ID3.get_label(row, tree)
        label = 1 if label == 'yes' else -1
        avg += label
        row_predictions.append(label)
    avg /= len(row_predictions)

    # calculate bias, variance
    y = 1 if row['y'] == 'yes' else -1
    single_bias = pow(y - avg, 2)
    single_var = np.var(row_predictions)

    # update totol bias, variance
    sum_single_bias += single_bias
    sum_single_var += single_var

avg_single_bias = sum_single_bias/len(Data_test)
avg_single_var = sum_single_var/len(Data_test)

print("single tree bias:", avg_single_bias, "\nsingle tree var:", avg_single_var, "\ngeneral error:", avg_single_bias+avg_single_var)

# calculate random forest bias and variance
sum_forest_bias = 0.0
sum_forest_var = 0.0
for row in Data_test:
    avg = 0
    row_predictions = []
    # calculate individual and mean predictions for each forest
    for trees in predictors:
        label = RandomForest.get_label_forest(row, trees)
        label /= len(trees)
        avg += label
        row_predictions.append(label)
    avg /= len(row_predictions)

    # calculate bias, variance
    y = 1 if row['y'] == 'yes' else -1
    forest_bias = pow(y - avg, 2)
    forest_var = np.var(row_predictions)

    # update totol bias, variance
    sum_forest_bias += forest_bias
    sum_forest_var += forest_var

avg_forest_bias = sum_forest_bias/len(Data_test)
avg_forest_var = sum_forest_var/len(Data_test)

print("forest bias:", avg_forest_bias, "\nforest var:", avg_forest_var, "\ngeneral error:", avg_forest_bias+avg_forest_var)

# single tree bias: 0.4117951200000014 
# single tree var: 0.1878448799999965 
# general error: 0.599639999999998
# forest bias: 0.4196513661391984 
# forest var: 0.0010428757487999998 
# general error: 0.4206942418879984