#  Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance
import random
import numpy as np
import matplotlib.pyplot as plt
import mnist
import pdb

def sqDistance(p, q, pSOS, qSOS):
    #  Efficiently compute squared euclidean distances between sets of vectors

    #  Compute the squared Euclidean distances between every d-dimensional point
    #  in p to every d-dimensional point in q. Both p and q are
    #  npoints-by-ndimensions.
    #  d(i, j) = sum((p(i, :) - q(j, :)).^2)

    d = np.add(pSOS, qSOS.T) - 2*np.dot(p, q.T)
    return d

np.random.seed(1)


def optical_character_recognition(train_size):
    Xtrain, ytrain, Xtest, ytest = mnist.load_data()
    # train_size = 10000
    test_size  = 10000

    Xtrain = Xtrain[0:train_size]
    ytrain = ytrain[0:train_size]

    Xtest = Xtest[0:test_size]
    ytest = ytest[0:test_size]

    #  Precompute sum of squares term for speed
    XtrainSOS = np.sum(Xtrain**2, axis=1, keepdims=True)
    XtestSOS  = np.sum(Xtest**2, axis=1, keepdims=True)

    #  fully solution takes too much memory so we will classify in batches
    #  nbatches must be an even divisor of test_size, increase if you run out of memory 
    if test_size > 1000:
      nbatches = 50
    else:
      nbatches = 5

    batches = np.array_split(np.arange(test_size), nbatches)
    ypred = np.zeros_like(ytest)

    #  Classify
    for i in range(nbatches):
        dst = sqDistance(Xtest[batches[i]], Xtrain, XtestSOS[batches[i]], XtrainSOS)
        closest = np.argmin(dst, axis=1)
        ypred[batches[i]] = ytrain[closest]

    #  Report
    errorRate = (ypred != ytest).mean()
    print('Error Rate: {:.2f}%\n'.format(100*errorRate))

    return errorRate
    #  image plot
    # plt.imshow(Xtrain[0].reshape(28, 28), cmap='gray')
    # plt.show()

def q1():
    # Plot a figure where the x-axis is number of training
    # examples (e.g. 100, 1000, 2500, 5000, 7500, 10000), and the y-axis is test error.

    train_size_set = [100, 1000, 2500, 5000, 7500, 10000]
    test_error = [0] * len(train_size_set)
    for i in range(len(train_size_set)):
        test_error[i] = optical_character_recognition(train_size_set[i])

    for a,b in zip(train_size_set, test_error): 
        plt.text(a, b, str(b))
    plt.plot(train_size_set, test_error, 'r', label='test-accuracy')
    plt.xlabel('Number of training samples') 
    plt.ylabel('test error')
    plt.xticks(train_size_set)

    plt.title('Number of training samples vs test error')
    plt.show()

q1()

def nfold_cross_validation(k, train_size):
    n = train_size// k
    Xtrain, ytrain, Xtest, ytest = mnist.load_data()
    Xtrain = Xtrain[0:train_size]
    ytrain = ytrain[0:train_size]
    indices = list(range(train_size))
    np.random.shuffle(indices)
    partitions = [indices[i * n:(i + 1) * n] for i in range((len(indices) + n - 1) // n )]
    if (n != (train_size / k)):     #leave out the last partition
        del partitions[-1]
    
    error = 0.0
    for i in range(len(partitions)):
        validation_idx = partitions[i]
        training_idx = list(set(indices) - set(partitions[i]))
        X_training, X_validation = Xtrain[training_idx], Xtrain[validation_idx]
        y_training, y_validation = ytrain[training_idx], ytrain[validation_idx]

        #  Precompute sum of squares term for speed
        X_training_SOS = np.sum(X_training**2, axis=1, keepdims=True)
        X_validation_SOS  = np.sum(X_validation**2, axis=1, keepdims=True)
        ypred = np.zeros_like(y_validation)

        #  Classify
        dst = sqDistance(X_validation, X_training, X_validation_SOS, X_training_SOS)
        closest = np.argmin(dst, axis=1)
        ypred = y_training[closest]

        #  Report
        errorRate = (ypred != y_validation).mean()
        error = error + errorRate
    
    return(error/ len(partitions))

nfolds = [3, 10, 50, 100, 1000]
cross_validation_error = [0] * len(nfolds)
for j in range(len(nfolds)):
    cross_validation_error[j] = nfold_cross_validation(nfolds[j], 1000)

for a,b in zip(nfolds, cross_validation_error): 
        plt.text(a, b, str(b))
plt.plot(nfolds, cross_validation_error, 'r', label='cross-validation-error')
plt.xlabel('Number of folds') 
plt.ylabel('Cross validation error')
plt.xticks(nfolds)

plt.title('n-folds cross validation error')
plt.show()