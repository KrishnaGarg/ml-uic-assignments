import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pdb
import math

def predict(X, w):
    n_ts = X.shape[0]
    # use w for prediction
    y=np.zeros(n_ts)
    pred = np.zeros(n_ts)       # initialize prediction vector
    wx = X @ w.reshape(-1, 1)
    for i in range(len(wx)):
        if wx[i] > 300: wx[i] = 1           # hack for nan values
        elif wx[i] < -300: wx[i] = 0
        y[i] = 1 / (1 + np.exp(wx[i]))
        if y[i] < 0.5:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred

def accuracy(X, y, w):
    count = 0
    y_pred = predict(X, w)
    for i in range(X.shape[0]):
        if y_pred[i] == y[i]:
            count += 1
    
    return count/X.shape[0]

def logistic_reg(X_tr, X_ts, y_tr, y_ts, lr):
    #perform gradient descent
    n_vars = X_tr.shape[1]  # number of variables
    n_tr = X_tr.shape[0]    # number of training examples
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.10        # tolerance for stopping

    iter = 0                # iteration counter
    max_iter = 1000         # maximum iteration
    test_accuracy = []
    train_accuracy = []
    iterations = []
    while (True):
        iter += 1

        # calculate gradient
        grad = np.zeros(n_vars) # initialize gradient
        
        wx = np.dot(X_tr, w.reshape(-1, 1))
        for i in range(len(wx)):
            if wx[i] > 300: wx[i] = 1           # hack for nan values
            elif wx[i] < -300: wx[i] = 0

        temp = np.exp(wx)/ (1 + np.exp(wx))
        grad = grad + np.dot((y_tr - temp[:,0]),X_tr)

        # for i in range(n_tr):
        #     wx = np.dot(w, X_tr[i])
        #     # if wx > 300: wx = 0.00001
        #     # elif wx < -300: wx = 0.00001
        #     temp = np.exp(wx)/ (1 + np.exp(wx))
        #     # print (temp)
        #     for j in range(n_vars):
        #         if (y_tr[i] - temp)*X_tr[i][j] > 10 or (y_tr[i] - temp)*X_tr[i][j] < -10 :
        #             continue
        #         else:
        #             grad[j] = grad[j] + (y_tr[i] - temp)*X_tr[i][j]

        w_new = w + lr * grad

        if iter%50 == 0:
            print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))
            # print(grad)
            iterations.append(iter)
            test_accuracy.append(accuracy(X_ts, y_ts, w_new))
            train_accuracy.append(accuracy(X_tr, y_tr, w_new))

        # stopping criteria and perform update if not stopping
        if (np.mean(np.abs(grad)) < tolerance):
            w = w_new
            break
        else :
            w = w_new
            
        if (iter >= max_iter):
            break

    plt.plot(iterations, test_accuracy, 'g', label='test-accuracy')
    plt.plot(iterations, train_accuracy, 'r', label='train-accuracy')
    plt.xlabel('Number of iterations') 
    plt.ylabel('accuracy')
    plt.xticks(iterations)
    plt.gca().legend(('test-accuracy', 'train-accuracy'))
    plt.title('#iterations vs accuracy for learning rate %f' %lr)
    plt.show()
    return test_accuracy[-1], train_accuracy[-1]

def regularized_logistic_reg(X_tr, X_ts, y_tr, y_ts, lr, regularize_factor):
    #perform gradient descent
    n_vars = X_tr.shape[1]  # number of variables
    n_tr = X_tr.shape[0]    # number of training examples
    w = np.zeros(n_vars)    # initialize parameter w
    tolerance = 0.10        # tolerance for stopping

    iter = 0                # iteration counter
    max_iter = 1000         # maximum iteration
    test_accuracy = []
    train_accuracy = []
    iterations = []
    while (True):
        iter += 1

        # calculate gradient
        grad = np.zeros(n_vars) # initialize gradient
        wx = np.dot(X_tr, w.reshape(-1, 1))
        for i in range(len(wx)):
            if wx[i] > 300: wx[i] = 1           # hack for nan values
            elif wx[i] < -300: wx[i] = 0
        temp = np.exp(wx)/ (1 + np.exp(wx))
        grad = grad + np.dot((y_tr - temp[:,0]),X_tr) + regularize_factor*w

        w_new = w + lr * grad

        if iter%50 == 0:
            print('Iteration: {0}, mean abs gradient = {1}'.format(iter, np.mean(np.abs(grad))))
            iterations.append(iter)
            test_accuracy.append(accuracy(X_ts, y_ts, w_new))
            train_accuracy.append(accuracy(X_tr, y_tr, w_new))

        # stopping criteria and perform update if not stopping
        if (np.mean(np.abs(grad)) < tolerance):
            w = w_new
            break
        else :
            w = w_new
            
        if (iter >= max_iter):
            break

    return test_accuracy[-1], train_accuracy[-1]

# read files
D_tr = genfromtxt('spambasetrain.csv', delimiter = ',', encoding = 'utf-8')
D_ts = genfromtxt('spambasetest.csv', delimiter = ',', encoding = 'utf-8')

# construct x and y for training and testing
X_tr = D_tr[: ,: -1]
y_tr = D_tr[: , -1]
X_ts = D_ts[: ,: -1]
y_ts = D_ts[: , -1]

# number of training / testing samples
n_tr = D_tr.shape[0]
n_ts = D_ts.shape[0]

# add 1 as feature
X_tr = np.concatenate((np.ones((n_tr, 1)), X_tr), axis = 1)
X_ts = np.concatenate((np.ones((n_ts, 1)), X_ts), axis = 1)

# set learning rate
lr = [1, 1e-2, 1e-4, 1e-6]
# for i in range(len(lr)):
#     test_accuracy, train_accuracy = logistic_reg(X_tr, X_ts, y_tr, y_ts, lr[i])
#     print('learning rate = {0}, train accuracy = {1}, test_accuracy = {2}'.format(str(lr[i]), str(train_accuracy), str(test_accuracy)))

regularize_factors = [0, pow(2,-8), pow(2,-6), pow(2,-4), pow(2,-2), 1, pow(2,2)]

lr2 = [1e-3]
regularize_test_accuracy = [0] * len(regularize_factors)
regularize_train_accuracy = [0] * len(regularize_factors)
for j in range(len(regularize_factors)):
        regularize_test_accuracy[j], regularize_train_accuracy[j] = regularized_logistic_reg(X_tr, X_ts, y_tr, y_ts, lr2, regularize_factors[j])
        print('lambda = {0}, learning rate = {1}, train accuracy = {2}, test_accuracy = {3}'.format(str(regularize_factors[j]), str(lr2), str(regularize_train_accuracy[j]), str(regularize_test_accuracy[j])))
    
plt.plot(regularize_factors, regularize_test_accuracy, 'g', label='regularized-test-accuracy')
plt.plot(regularize_factors, regularize_train_accuracy, 'r', label='regularized-train-accuracy')
plt.xlabel('lambda') 
plt.ylabel('accuracy')
plt.gca().legend(('test-accuracy', 'train-accuracy'))
plt.title('Accuracy after regularization')
plt.show()