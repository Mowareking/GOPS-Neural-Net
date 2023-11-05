# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip
import csv

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((13, 1))
    e[j] = 1.0
    return e

def load_gops():
    with open("training_data.csv") as raw_data:
        data = csv.reader(raw_data)
        training_inputs = [np.reshape(np.array(list(map(int, line[1:]))), (40, 1)) for line in data]

    with open("training_data.csv") as raw_data:
        data = csv.reader(raw_data)
        training_results = []
        for line in data:
            index = []
            opp_move = int(line[0])
            higher = sum(list(map(int, line[opp_move+1:14])))
            for j, digit in enumerate(line[1:14]):
                if int(digit):
                    if not higher:
                        index = j
                        break
                    else:
                        if j >= opp_move:
                            index = j
                            break
            training_results.append(vectorized_result(index))
    training_data = zip(training_inputs, training_results)

    with open("test_data.csv") as raw_data:
        data = csv.reader(raw_data)
        test_inputs = [np.reshape(np.array(list(map(int, line[1:]))), (40, 1)) for line in data]

    with open("test_data.csv") as raw_data:
        data = csv.reader(raw_data)
        test_results = []
        opp_move = int(line[0])
        for line in data:
            index = []
            higher = sum(list(map(int, line[opp_move+1:14])))
            for j, digit in enumerate(line[1:14]):
                if int(digit):
                    if not higher:
                        index = j
                        break
                    else:
                        if j >= opp_move:
                            index = j
                            break
            test_results.append(index)
    training_data = zip(training_inputs, training_results)
    test_data = zip(test_inputs, test_results)
    return training_data, test_data



