import os
import logging
import shutil
from os.path import join

from scipy.sparse import csr_matrix
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from caffe.proto.caffe_pb2 import SparseDatum
import caffe

import leveldb
from time import time
from sklearn.linear_model import SGDClassifier


def sparse_datum_generator(X, y):
    assert (isinstance(X, csr_matrix))

    for i in xrange(X.shape[0]):
        if i % 1000 == 0:
            print 'processed {} rows'.format(i)
        row = X[i]
        datum = SparseDatum()

        for k in xrange(len(row.data)):
            datum.data.append(float(row.data[k]))
            datum.indices.append(int(row.indices[k]))
        datum.nnz = int(row.nnz)
        datum.size = X.shape[1]
        datum.label = int(y[i])
        assert len(row.indptr) == 2
        assert row.indptr[0] == 0
        assert row.indptr[1] == row.nnz

        yield str(i), datum.SerializeToString()


def create_leveldb(X, y, data_folder, dbname):
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    folder = join(data_folder, dbname)
    if not os.path.isdir(data_folder):
        shutil.rmtree(folder)
    db = leveldb.LevelDB(folder)
    for id, value in sparse_datum_generator(X, y):
        db.Put(id, value)


def learn_and_test(solver_file, size_test):
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_file)
    solver.solve()

    accuracy = 0
    test_iters = int(size_test / solver.test_nets[0].blobs['data'].num)
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
    accuracy /= test_iters
    return accuracy


def main():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    # Uncomment the following to do the analysis on all the categories
    # categories = None

    print "Loading 20 newsgroups dataset for categories:"
    print categories if categories else "all"

    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42)

    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42)
    print 'data loaded'

    categories = data_train.target_names  # for case categories == None

    print "%d documents (training set)" % len(data_train.data)
    print "%d documents (testing set)" % len(data_test.data)
    print "%d categories" % len(categories)
    print

    # split a training set and a test set
    y_train, y_test = data_train.target, data_test.target

    print "Extracting features from the training dataset using a sparse vectorizer"
    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
    print "done in %fs" % (time() - t0)
    print "n_samples: %d, n_features: %d" % X_train.shape
    print

    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    print "done in %fs" % (time() - t0)
    print "n_samples: %d, n_features: %d" % X_test.shape
    print 'type test: {}'.format(type(X_test))

    # shuffle data ################################
    indices = range(X_train.shape[0])
    import random
    random.shuffle(indices)
    X_train = X_train[indices]
    y_train = [y_train[i] for i in indices]
    ###########################################

    create_leveldb(X_train, y_train, 'sparse_classification/data','sparse_train_leveldb')
    create_leveldb(X_test, y_test, 'sparse_classification/data', 'sparse_test_leveldb')

    acc = learn_and_test('sparse_classification/solver.prototxt', X_test.shape[0])
    print("Accuracy: {:.3f}".format(acc))

    acc = learn_and_test('sparse_classification/solver2.prototxt', X_test.shape[0])
    print("Accuracy second model: {:.3f}".format(acc))

    clf = SGDClassifier(
    loss='log', n_iter=1000, penalty='l2', alpha=1e-3, class_weight='auto')

    clf.fit(X_train, y_train)
    yt_pred = clf.predict(X_test)
    print('Accuracy scikit learn: {:.3f}'.format(sklearn.metrics.accuracy_score(y_test, yt_pred)))


if __name__ == "__main__":
    main()