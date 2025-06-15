import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    for epoch in range(MAX_ITERS):
        grad_w = [0.0] * numvars
        grad_b = 0.0

        # Loop over all training examples
        for x, y in data:
            dot_product = sum([w[i] * x[i] for i in range(len(w))]) + b
            prob = 1.0 / (1.0 + exp(-(y * dot_product)))
            factor = (1 - prob) * y

            for j in range(numvars):
                grad_w[j] += factor * (-x[j])
            grad_b += -factor
        
        #l2 regularization
        for j in range(numvars):
            grad_w[j] += l2_reg_weight * w[j]

        magnitude = sqrt(sum(grad_w[i] ** 2 for i in range(numvars)) + grad_b ** 2)
        if magnitude < 0.0001:
            break
        
        for j in range(numvars):
            w[j] -= eta * grad_w[j]
        b -= eta * grad_b

    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model

    dot_product = 0.0
    dot_product += sum([w[i] * x[i] for i in range(len(w))])
    dot_product += b
    prob = 1.0 / (1.0 + exp(-dot_product))

    return prob  # This is an random probability, fix this according to your solution


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
