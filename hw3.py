from numpy import *
import numpy.random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import bisect
import sys
import os

dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
tests = data[idx[10000:], :].astype(int)
tests_labels = labels[idx[10000:]]


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

def a_part():
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
    fig.savefig(os.path.join(dir_path, 'a_c.png'))
    fig.clf()

    best_k = accurate_rates.index(max(accurate_rates)) + 1
    print "The k with the best results is %d" % best_k
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
    fig.savefig(os.path.join(dir_path, 'a_d.png'))
    fig.clf()

a_part()