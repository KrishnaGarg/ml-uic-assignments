import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle
from scipy.spatial.distance import cdist
from scipy import stats
import statistics
from statistics import mode
import pdb

#  KNN function
def knn_predict(X_test, X_train_temp, y_train_temp, k=1):
    n_X_test = X_test.shape[0]
    decision = np.zeros((n_X_test, 1))
    for i in range(n_X_test):
        point = X_test[[i],:]

        #  compute euclidan distance from the point to all training data
        dist = cdist(X_train_temp, point)
        #  sort the distance, get the index
        idx_sorted = np.argsort(dist, axis=0)

        #  find the most frequent class among the k nearest neighbour
        pred = stats.mode( y_train_temp[idx_sorted[0:k]] )

        decision[i] = pred[0]
    return decision

np.random.seed(1)

# Setup data
D = np.genfromtxt('iris.csv', delimiter=',')
X_train = D[:, 0:2]   # feature
y_train = D[:, -1]    # label

    # Setup meshgrid
x1, x2 = np.meshgrid(np.arange(2,5,0.01), np.arange(0,3,0.01))
X12 = np.c_[x1.ravel(), x2.ravel()]

def createDatasetWithOutliers(m):
    indices = list(range(m))
    np.random.shuffle(indices)
    
    X_train_new = X_train
    y_train_new = y_train
    for i in range(len(indices)):
        if y_train_new[indices[i]] == 1:
            y_train_new[indices[i]] = random.choice([2,3])
        elif y_train_new[indices[i]] == 2:
            y_train_new[indices[i]] = random.choice([1,3])
        else:
            y_train_new[indices[i]] = random.choice([1,2])
    return X_train_new, y_train_new

np.random.seed(1)

def leave_one_out_cross_validation(Xtrain, ytrain, n):
    indices = list(range(n))
    np.random.shuffle(indices)
    error = 0.0
    for i in range(n):
        validation_idx = [i]
        training_idx = list(set(indices) - set([i]))
        
        X_training, X_validation = Xtrain[training_idx], Xtrain[validation_idx]
        y_training, y_validation = ytrain[training_idx], ytrain[validation_idx]

        ypred = knn_predict(X_validation, X_training, y_training, 3)
        errorRate = (ypred != y_validation)        
        error = error + errorRate
    
    return(error/ n)

outliers = [10, 20, 30, 50]
X_train_new = [0] * 4
y_train_new = [0] * 4

for i in range(4):
    X_train_new[i], y_train_new[i] = createDatasetWithOutliers(outliers[i])
    error = leave_one_out_cross_validation(X_train_new[i], y_train_new[i], X_train_new[i].shape[0])
    print("LOOCV error rate for %d outliers = %3.4f" %(outliers[i], error))
    decision = knn_predict(X12, X_train_new[i], y_train_new[i], 3)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    #  plot decisions in the grid
    decision = decision.reshape(x1.shape)
    plt.figure()
    plt.pcolormesh(x1, x2, decision, cmap=cmap_light)

    # Plot the training points
    plt.scatter(X_train_new[i][:, 0], X_train_new[i][:, 1], c=y_train_new[i], cmap=cmap_bold, s=25)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.title('Plot for robustness against %d outliers' %outliers[i])
    plt.show()