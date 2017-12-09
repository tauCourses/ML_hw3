import math
import bisect
import sys
import os
from numpy import *
import numpy.random
from scipy.stats import mode
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from sklearn import svm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']


# k nearest nneighbors
def knn(images, labels, query_image, k, _labels=None, _nearest=None):
    k_labels = _labels if _labels is not None else []
    k_nearest = _nearest if _nearest is not None else []

    for image, label in zip(images, labels):
        dis = numpy.linalg.norm(image - query_image)
        inx = bisect.bisect(k_nearest, dis)
        if inx < k:
            k_nearest.insert(inx, dis)
            k_labels.insert(inx, label)

    return k_labels, k_nearest


def get_mode_value(test_list, k):
    return mode(test_list[:k])[0][0]


def check_test_data(test_lists, tests_labels, k):
    succeeded = [True for test_list, label in zip(test_lists, tests_labels) if get_mode_value(test_list, k) == label]
    return float(len(succeeded)) / len(tests_labels)

def assignment_1():
    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    tests = data[idx[10000:], :].astype(int)
    tests_labels = labels[idx[10000:]]

    ks = [i for i in xrange(1,101,1)]
    _train_images = train[:1000]
    _train_labels = train_labels[:1000]

    labels_lists = []
    for test in tests:  # pre calculation of the distances for each image with all train images
        labels_lists.append(knn(_train_images, _train_labels, test, max(ks))[0])

    print "With k=10 the accurate rate is %f" % check_test_data(labels_lists, tests_labels, 10)

    accurate_rates = []
    for k in ks:
        accurate_rates.append(check_test_data(labels_lists, tests_labels, k))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ks, accurate_rates)
    plt.xlabel('k', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xticks(arange(0, 101, 5))
    fig.savefig(os.path.join(dir_path, '1_c.png'))
    fig.clf()

    best_k = accurate_rates.index(max(accurate_rates)) + 1
    print "The k with the highest accuracy rate is %d" % best_k
    sizes = [i for i in xrange(100,5100,100)]
    test_lists = [([],[]) for x in xrange(len(tests))]

    accurate_rates = []
    for ts in sizes: # ts = train size
        succeess_counter = 0
        for i, test in enumerate(tests):
            test_lists[i] = knn(train[ts-100:ts], train_labels[ts-100:ts], test, best_k, test_lists[i][0], test_lists[i][1])
            succeess_counter += 1 if get_mode_value(test_lists[i][0], best_k) == tests_labels[i] else 0
        accurate_rates.append(float(succeess_counter)/len(tests))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sizes, accurate_rates)
    plt.xlabel('Tarining set size', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xticks(arange(0, 5100, 500))
    fig.savefig(os.path.join(dir_path, '1_d.png'))
    fig.clf()

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

test_data_unscaled = data[60000 + test_idx, :].astype(float)
test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

# Pre-processing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


# Perceptron
def create_perceptron_classifier(train_data, train_labels):
    w = numpy.zeros_like(train_data[0])
    for data, label in zip(train_data, train_labels):
        dot_product = numpy.dot(w, data)
        predicted_label = (dot_product > 0) * 2 - 1
        if predicted_label != label:
            w += (data / numpy.linalg.norm(data)) * label
    w /= numpy.linalg.norm(w)  # normalize
    return (lambda x: (numpy.dot(w, x) > 0) * 2 - 1), w


def test_perceptron(test_data, test_labels, classifier):
    accuracy = len([True for data, label in zip(test_data,test_labels) if classifier(data) == label])
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
    ax.fill_between(ns, percentiles95, percentiles5, facecolors="#3F5D7D")
    ax.plot(ns, means, '.r-', label='Mean')
    ax.plot(ns, percentiles95, '.g-', label='Percentiles 95')
    ax.plot(ns, percentiles5, '.b-', label='Percentiles 5')
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
    print("The Perceptron algorithm accuracy using all the train data in the test data was: " + str(accuracy))


def assignment_2_d():
    classifier, _ = create_perceptron_classifier(train_data, train_labels)
    false_examples = [unscaled for data, label, unscaled in zip(test_data, test_labels, test_data_unscaled) if classifier(data) != label]

    false_examples = false_examples[:2]
    plt.imshow(reshape(false_examples[0], (28, 28)), interpolation="nearest")
    plt.savefig(os.path.join(dir_path, '2_d_false_example_1.png'))
    plt.imshow(reshape(false_examples[1], (28, 28)), interpolation="nearest")
    plt.savefig(os.path.join(dir_path, '2_d_false_example_2.png'))


def assignment_2():
    assignment_2_a()
    assignment_2_b()
    assignment_2_c()
    assignment_2_d()


# SVM
def assignment_3_a():
    c_options = [math.pow(10, i) for i in numpy.linspace(-10, 10, 100)]
    train_result_accuracy = []
    validation_result_accuracy = []
    for c in c_options:
        svc = svm.LinearSVC(loss='hinge', fit_intercept=False, C=c)
        svc.fit(train_data, train_labels)
        train_accuracy = svc.score(train_data, train_labels)
        validation_accuracy = svc.score(validation_data, validation_labels)
        train_result_accuracy.append(train_accuracy)
        validation_result_accuracy.append(validation_accuracy)
    max_value = max(validation_result_accuracy)
    max_index = validation_result_accuracy.index(max_value)
    my_best_c = c_options[max_index]
    print("Best c for LinearSVC was: " + str(my_best_c))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.plot(c_options, train_result_accuracy, 'r-', label='Train Accuracy', )
    ax.plot(c_options, validation_result_accuracy, 'b-', label='Validation Accuracy')
    plt.legend()
    plt.xlabel('C', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    fig.savefig(os.path.join(dir_path, '3_a.png'))
    fig.clf()
    return my_best_c


def assignment_3_c(my_best_c):
    svc = svm.LinearSVC(loss='hinge', fit_intercept=False, C=my_best_c)
    svc.fit(train_data, train_labels)
    plt.imshow(reshape(svc.coef_, (28, 28)), interpolation="nearest")
    plt.savefig(os.path.join(dir_path, '3_c.png'))


def assignment_3_d(my_best_c):
    svc = svm.LinearSVC(loss='hinge', fit_intercept=False, C=my_best_c)
    svc.fit(train_data, train_labels)
    test_accuracy = svc.score(test_data, test_labels)
    print("LinearSVC with best C had accuracy rate of " + str(test_accuracy))


def assignment_3_e():
    c = 10
    gamma = 5 * pow(10, -7)
    svc = svm.SVC(C=c, gamma=gamma)
    svc.fit(train_data, train_labels)
    train_accuracy = svc.score(train_data, train_labels)
    test_accuracy = svc.score(test_data, test_labels)
    print("RBF SVC kernel with best C=10 and gamma=5*1e-7 had accuracy rate of " + str(
        train_accuracy) + " on the training set, and " + str(test_accuracy) + " on the test set")

def assignment_3():
    best_c = assignment_3_a()
    assignment_3_c(best_c)
    assignment_3_d(best_c)
    assignment_3_e()

if len(sys.argv) < 2:
    print "Please enter which part do you want to execute - 1, 2, 3 or all"
    exit()
cmds = sys.argv[1:]
for cmd in cmds:
    if cmd not in ['1', '2', '3', 'all']:
        print "Unknown argument %s. please run with 1, 2, 3 or all" % cmd
        exit()

if '1' in cmds or 'all' in cmds:
    assignment_1()
if '2' in cmds or 'all' in cmds:
    assignment_2()
if '3' in cmds or 'all' in cmds:
    assignment_3()
