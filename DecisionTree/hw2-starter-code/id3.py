import sys
import re
# Node class for the decision tree
import node
import math

train = None
varnames = None
test = None
testvarnames = None
root = None

# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    if p == 0 or p == 1:
        return 0
    return (-p*math.log2(p))-((1-p)*math.log2(1-p))

# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    if pxi == 0 or pxi == total:
        return 0

    # Step 1: Compute entropy of the whole dataset before the split
    entropy_before_split = entropy(py / total)

    #entropy for xi =1
    Hx1 = entropy(py_pxi / pxi)

    #entropy for xi = 0
    Hx0 = entropy((py - py_pxi) / (total - pxi))

    #entropy after split
    weighted_entropy = ((pxi / total) * Hx1) + (((total - pxi) / total) * Hx0)

    #compute information gain
    info_gain = entropy_before_split - weighted_entropy

    return info_gain

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
def count_class_label(data, var_index, class_index):
    pxi = 0
    py = 0
    py_pxi = 0
    total = len(data)

    for row in data:
        if row[var_index] == 1:
            pxi += 1
            if row[class_index] == 1:
                py_pxi += 1
        if row[class_index] == 1:
            py += 1

    return (py_pxi, pxi, py, total)
        
# - find the best variable to split on, according to mutual information
def best_Split(data):
    best_gain = -float('inf')
    best_feature = None
    class_index = len(data[0]) - 1

    for index in range(class_index):
        py_pxi, pxi, py, total = count_class_label(data, index, class_index)
        gain = infogain(py_pxi, pxi, py, total)
        if gain > best_gain:
            best_gain = gain
            best_feature = index
    return best_feature

# Partition data based on a given variable
def partition_data(data, split_index):
    group_one = [row for row in data if row[split_index] == 1]
    group_zero = [row for row in data if row[split_index] == 0]
    return group_one, group_zero

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    class_index = len(data[0]) - 1
    labels = [row[class_index] for row in data]

    # Base case: all examples have the same label
    if all(label == labels[0] for label in labels):
        return node.Leaf(varnames, labels[0])

    # Base case: no variables left to split
    if len(data[0]) == 1:  # Only class label remains
        majority_label = max(set(labels), key=labels.count)
        return node.Leaf(varnames, majority_label)

    # Recursive case
    best_var = best_Split(data)

    # If no useful split can be found (gain is zero), return majority label
    if best_var is None:
        majority_label = max(set(labels), key=labels.count)
        return node.Leaf(varnames, majority_label)

    data1, data0 = partition_data(data, best_var)

    # If one split is empty, avoid further recursion
    if not data1 or not data0:
        majority_label = max(set(labels), key=labels.count)
        return node.Leaf(varnames, majority_label)

    left_subtree = build_tree(data0, varnames)
    right_subtree = build_tree(data1, varnames)

    return node.Split(varnames, best_var, left_subtree, right_subtree)

# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
