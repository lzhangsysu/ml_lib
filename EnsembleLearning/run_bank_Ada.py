import statistics
import copy
import ID3
import AdaBoost
import matplotlib.pyplot as plt

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


# Q2.a
# Run AdaBoosting for T = 1-500, plot error-T relationship
outFile = open("ada_out.txt", 'w')
outFile.write("iter\terr_train\terr_test\n")
for T in [1,2,3,4,5,6,8,10,15,20,30,50,70,90,120,150,200,250,300,350,400,450,500]:
    trees, alphas = AdaBoost.AdaBoost_train(Data_train, Attributes, Labels, T)
    hit_train = AdaBoost.AdaBoost_test(Data_train, trees, alphas)
    hit_test = AdaBoost.AdaBoost_test(Data_test, trees, alphas)
    outFile.write(str(T) + "\t" + str(1-hit_train) + "\t" + str(1-hit_test) + "\n")
    reset_weight(Data_train)

outFile.close()


# for T=500, find training and test error in each iteration
e_t, e_r = AdaBoost.print_err_Ada(Data_train, Data_test, Attributes, Labels, 500)
t = [i+1 for i in range(0,500)]

fig, ax = plt.subplots(figsize = (6,4))
ax.plot(t, e_r, label='test error', c='grey', alpha=0.3)
ax.plot(t, e_t, label='training error')
ax.legend()
ax.set_title("Error per iteration for Adaboost")
ax.set_xlabel('iteration')
ax.set_ylabel('error')

plt.show()

