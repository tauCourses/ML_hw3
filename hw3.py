from numpy import *
import numpy.random
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import bisect

mnist = fetch_mldata('MNIST original')
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


def knn(images, labels, query_image, k):
    k_nearest = []
    k_labels = []
    for image, label in zip(images, labels):
        dis = numpy.linalg.norm(image - query_image)
        inx = bisect.bisect(k_nearest, dis)
        if inx < k:
            k_nearest.insert(inx, dis)
            k_labels.insert(inx, label)

    return mode(k_labels[:k])[0][0]


def check_test_data(train_data, train_labels, test_data, test_lables, k):
    return len([True for query_image, label in zip(test_data, test_lables) if
                knn(train_data, train_labels, query_image, k) != label])


ks = [i + 1 for i in range(100)]
errors = []
_train_images = train_data[:1000]
_train_labels = train_labels[:1000]

for k in ks:
    print(k)
    errors.append(check_test_data(_train_images, _train_labels, test_data, test_labels, k))

plt.plot(ks, errors)
plt.show()
print(errors)
