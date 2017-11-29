from numpy import *
import os
import sys
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import pickle
import matplotlib.pyplot as plt

dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")

# mnist = fetch_mldata('MNIST original')
# with open('mnist.pkl', 'wb') as output:
#     pickle.dump(mnist, output, pickle.HIGHEST_PROTOCOL)
with open('mnist.pkl', 'rb') as input:
    mnist = pickle.load(input)

data = mnist['data']
labels = mnist['target']

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

test_data_unscaled = data[60000 + test_idx, :].astype(float)
test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


# Perceptron
def create_perceptron_classifier(train_data, train_labels):
    w = numpy.zeros_like(train_data[0])
    for i in range(len(train_data)):
        dot_product = numpy.dot(w, train_data[i])
        predicted_label = (dot_product > 0) * 2 - 1
        if predicted_label != train_labels[i]:
            w += (train_data[i] / numpy.linalg.norm(train_data[i])) * train_labels[i]
    w /= numpy.linalg.norm(w)  # normalize
    return (lambda x: (numpy.dot(w, x) > 0) * 2 - 1), w


def test_perceptron(test_data, test_labels, classifier):
    accuracy = 0
    for i in range(len(test_data)):
        if classifier(test_data[i]) == test_labels[i]:
            accuracy += 1
    return float(accuracy) / len(test_data)


def get_random_train_data_and_labels_permutation(n):
    permutation = numpy.random.permutation(range(n))
    return [train_data[i] for i in permutation], [train_labels[i] for i in permutation]


def assignment_2_a():
    number_of_runs_per_n = 100
    ns = [5, 10, 50, 100, 500, 1000, 5000]
    accuracies = []
    for n in ns:
        accuracy = []
        for _ in range(number_of_runs_per_n):
            current_train_data, current_train_labels = get_random_train_data_and_labels_permutation(n)
            current_classifier, _ = create_perceptron_classifier(current_train_data, current_train_labels)
            current_accuracy = test_perceptron(test_data, test_labels, current_classifier)
            accuracy.append(current_accuracy)
        accuracies.append(accuracy)
    means = [numpy.mean(x) for x in accuracies]
    percentiles5 = [numpy.percentile(x, 5) for x in accuracies]
    percentiles95 = [numpy.percentile(x, 95) for x in accuracies]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.plot(ns, means, '.r-', label='Mean')
    ax.plot(ns, percentiles5, '.b-', label='Percentiles 5')
    ax.plot(ns, percentiles95, '.g-', label='Percentiles 95')
    plt.legend()
    plt.xlabel('train data size', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    fig.savefig(os.path.join(dir_path, '2_a.png'))
    fig.clf()


def assignment_2_b():
    _, w = create_perceptron_classifier(train_data, train_labels)
    plt.imshow(reshape(w, (28, 28)), interpolation="nearest")
    plt.savefig(os.path.join(dir_path, '2_b.png'))


def assignment_2_c():
    classifier, _ = create_perceptron_classifier(train_data, train_labels)
    accuracy = test_perceptron(test_data, test_labels, classifier)
    print("The Perstron algorithm accuracy using all the train data in the test data was: " + str(accuracy))


def assignment_2_d():
    classifier, _ = create_perceptron_classifier(train_data, train_labels)
    false_examples = []
    for i in range(len(test_data)):
        if classifier(test_data[i]) != test_labels[i]:
            false_examples.append(test_data_unscaled[i])
    false_examples = false_examples[:2]
    plt.imshow(reshape(false_examples[0], (28, 28)), interpolation="nearest")
    plt.savefig(os.path.join(dir_path, '2_d_false_example_1.png'))
    plt.imshow(reshape(false_examples[1], (28, 28)), interpolation="nearest")
    plt.savefig(os.path.join(dir_path, '2_d_false_example_2.png'))


assignment_2_a()
assignment_2_b()
assignment_2_c()
assignment_2_d()
