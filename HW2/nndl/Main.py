# Implement the function compute_distances() in the KNN class.
# Do not worry about the input 'norm' for now; use the default definition of the norm
# in the code, which is the 2-norm.
# You should only have to fill out the clearly marked sections.
import sys
from knn import KNN
import time
import numpy as np # for doing most of our calculations
import matplotlib.pyplot as plt# for plotting
sys.path.insert(1,'D:\\file_yyux\\UCLA\\23W-courses\\C247\\hw2_Questions\\code')
from utils.data_utils import load_CIFAR10 # function to load the CIFAR-10 dataset.

# Load matplotlib images inline
#% matplotlib inline

# These are important for reloading any code you write in external .py files.
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

# Set the path to the CIFAR-10 data
cifar10_dir = 'D:\\file_yyux\\UCLA\\23W-courses\\C247\\cifar-10-batches-py' # You need to update this line
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)#Training data shape:  (50000, 32, 32, 3)
print('Training labels shape: ', y_train.shape)#Training labels shape:  (50000,)
print('Test data shape: ', X_test.shape)#Test data shape:  (10000, 32, 32, 3)
print('Test labels shape: ', y_test.shape)#Test labels shape:  (10000,)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)#5000,500
#print(y_train.shape)#5000

'''
K-Nearest Neighbour

I. KNN Prediction
'''

# Declare an instance of the knn class.
knn = KNN()

# Train the classifier.
#   We have implemented the training of the KNN classifier.
#   Look at the train function in the KNN class to see what this does.
knn.train(X=X_train, y=y_train)

#time_start =time.time()

#dists_L2 = knn.compute_distances(X=X_test)
'''
print('Time to run code: {}'.format(time.time()-time_start))
print('Frobenius norm of L2 distances: {}'.format(np.linalg.norm(dists_L2, 'fro')))

'''

'''
II. KNN vectorization
'''

# Implement the function compute_L2_distances_vectorized() in the KNN class.
# In this function, you ought to achieve the same L2 distance but WITHOUT any for loops.
# Note, this is SPECIFIC for the L2 norm.

time_start =time.time()
dists_L2_vectorized = knn.compute_L2_distances_vectorized(X=X_test)
#print('Time to run code: {}'.format(time.time()-time_start))
#print('Difference in L2 distances between your KNN implementations (should be 0): {}'.format(np.linalg.norm(dists_L2 - dists_L2_vectorized, 'fro')))


'''
Implementing the prediction

Now that we have functions to calculate the distances from a test point to given training points, 
we now implement the function that will predict the test point labels.



If you implemented this correctly, the error should be: 0.726.

This means that the k-nearest neighbors classifier is right 27.4% of the time, 
which is not great, considering that chance levels are 10%.
'''

# Implement the function predict_labels in the KNN class.
# Calculate the training error (num_incorrect / total_samples)
#   from running knn.predict_labels with k=1

error = 1

# ================================================================ #
# YOUR CODE HERE:
#   Calculate the error rate by calling predict_labels on the test
#   data with k = 1.  Store the error rate in the variable error.
# ================================================================ #
y_pred = knn.predict_labels(dists_L2_vectorized, k=1)#/dists_L2.shape[0]
accuracy = np.mean(y_pred == y_test)
error = 1 - accuracy
pass
# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #

print(error)


'''
Optimizing KNN hyperparameters
In this section, we'll take the KNN classifier that you have constructed and 
perform cross-validation to choose a best value of  𝑘, as well as a best choice of norm.
'''
#Create training and validation folds
#First, we will create the training and validation folds for use in k-fold cross validation.
# Create the dataset folds for cross-valdiation.
num_folds = 5

X_train_folds = []
y_train_folds =  []

# ================================================================ #
# YOUR CODE HERE:
#   Split the training data into num_folds (i.e., 5) folds.
#   X_train_folds is a list, where X_train_folds[i] contains the
#      data points in fold i.
#   y_train_folds is also a list, where y_train_folds[i] contains
#      the corresponding labels for the data in X_train_folds[i]
# ================================================================ #
'''
#shuffle the datasets before splitting
permutation = np.random.permutation(5000)
shuffled_X = X_train[permutation,:]
shuffled_y = y_train[permutation]

#split into 5 folds
#n_sample = len(X_train)
#fold_size = n_sample // num_folds
X_train_folds = np.split(shuffled_X, num_folds)
y_train_folds = np.split(shuffled_y, num_folds)

X_tr = np.split(shuffled_X, num_folds)
y_tr = np.split(shuffled_y, num_folds)
for fold in range(num_folds):
    X_train_folds.append(X_tr[fold])
    y_train_folds.append(y_tr[fold])
#X_Train = np.concatenate((X_train_folds[0:3]), axis=0, out=None, dtype=None, casting="same_kind")
#X_Val = np.concatenate((X_train_folds[4]), axis=0, out=None, casting="same_kind")
'''
np.random.seed(0)
assert num_training == X_train.shape[0]
permutation = np.random.permutation(num_training)
for i in range(num_folds):
    start_id = int(i*num_training/num_folds)
    end_id = int((i+1)*num_training/num_folds)
    X_train_folds.append(X_train[permutation[start_id:end_id],:])
    y_train_folds.append(y_train[permutation[start_id:end_id]])

X_rst_folds = []
y_rst_folds = []

X_rst_folds.append( np.concatenate((X_train_folds[1], X_train_folds[2], X_train_folds[3], X_train_folds[4]), axis = 0) )
y_rst_folds.append( np.concatenate((y_train_folds[1], y_train_folds[2], y_train_folds[3], y_train_folds[4]), axis = 0) )

X_rst_folds.append( np.concatenate((X_train_folds[0], X_train_folds[2], X_train_folds[3], X_train_folds[4]), axis = 0) )
y_rst_folds.append( np.concatenate((y_train_folds[0], y_train_folds[2], y_train_folds[3], y_train_folds[4]), axis = 0) )

X_rst_folds.append( np.concatenate((X_train_folds[0], X_train_folds[1], X_train_folds[3], X_train_folds[4]), axis = 0) )
y_rst_folds.append( np.concatenate((y_train_folds[0], y_train_folds[1], y_train_folds[3], y_train_folds[4]), axis = 0) )

X_rst_folds.append( np.concatenate((X_train_folds[0], X_train_folds[2], X_train_folds[1], X_train_folds[4]), axis = 0) )
y_rst_folds.append( np.concatenate((y_train_folds[0], y_train_folds[2], y_train_folds[1], y_train_folds[4]), axis = 0) )

X_rst_folds.append( np.concatenate((X_train_folds[0], X_train_folds[2], X_train_folds[3], X_train_folds[1]), axis = 0) )
y_rst_folds.append( np.concatenate((y_train_folds[0], y_train_folds[2], y_train_folds[3], y_train_folds[1]), axis = 0) )


pass

# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #

'''
time_start =time.time()

ks = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

# ================================================================ #
# YOUR CODE HERE:
#   Calculate the cross-validation error for each k in ks, testing
#   the trained model on each of the 5 folds.  Average these errors
#   together and make a plot of k vs. cross-validation error. Since
#   we are assuming L2 distance here, please use the vectorized code!
#   Otherwise, you might be waiting a long time.
# ================================================================ #

####TEST
# ============================================================================= #
#er = []        TEST.part
#for k in ks[0:2]:
#    dis = []
#    yp = []
#    acc = []
#    for fold in range(num_folds):
#        dis.append(knn.compute_L2_distances_vectorized(X = X_train_folds[fold]))
#        yp.append(knn.predict_labels(dis[fold], k))
#        acc.append(np.mean(yp[fold] == y_train_folds[fold]))
#    er.append(1 - np.mean(acc))
#print(er)
# ============================================================================= #
cv_error = np.zeros((len(ks),num_folds))
for i, k in enumerate(ks):
    #dists_L2_vectorized = []
    #y_pred = []
    #accuracy = []
    for fold in range(num_folds):
        knn.train(X=X_rst_folds[fold], y=y_rst_folds[fold])
        dists_L2_vectorized = knn.compute_L2_distances_vectorized(X = X_train_folds[fold])
        y_pred = knn.predict_labels(dists_L2_vectorized, k)
        #accuracy = (np.mean(y_pred[fold] == y_train_folds[fold]))
        cv_error[i,fold] = sum(y_pred != y_train_folds[fold])/X_train_folds[fold].shape[0]
error_k = np.average(cv_error, axis = 1)
plt.plot(ks,error_k,'s-',color = 'r',label = '')
plt.xlabel("number of neignbors")
plt.ylabel("Cross-Validation Error")
plt.show()
print(error_k)
pass

# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #

print('Computation time: %.2f'%(time.time()-time_start))

'''

# =============================================================== #
# OPTIMIZING the NORM
# =============================================================== #
time_start =time.time()

L1_norm = lambda x: np.linalg.norm(x, ord=1)
L2_norm = lambda x: np.linalg.norm(x, ord=2)
Linf_norm = lambda x: np.linalg.norm(x, ord= np.inf)
norms = [L1_norm, L2_norm, Linf_norm]
'''
# ================================================================ #
# YOUR CODE HERE:
#   Calculate the cross-validation error for each norm in norms, testing
#   the trained model on each of the 5 folds.  Average these errors
#   together and make a plot of the norm used vs the cross-validation error
#   Use the best cross-validation k from the previous part.
#
#   Feel free to use the compute_distances function.  We're testing just
#   three norms, but be advised that this could still take some time.
#   You're welcome to write a vectorized form of the L1- and Linf- norms
#   to speed this up, but it is not necessary.
# ================================================================ #
cv_error = np.zeros((len(norms),num_folds))
for i, norm in enumerate(norms):
    for fold in range(num_folds):
        knn.train(X=X_rst_folds[fold], y=y_rst_folds[fold])
        dists_L2 = knn.compute_distances(X=X_train_folds[fold],norm = norm)
        y_pred = knn.predict_labels(dists_L2, k=5)
        cv_error[i, fold] = sum(y_pred != y_train_folds[fold]) / X_train_folds[fold].shape[0]
error_k = np.average(cv_error, axis=1)
n = np.arange(3)
plt.plot(n, error_k, 's-', color='r', label='')
plt.xlabel("order of norms")
plt.ylabel("Cross-Validation Error")
plt.show()
print(error_k)

pass

# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #
print('Computation time: %.2f'%(time.time()-time_start))
'''
error = 1

# ================================================================ #
# YOUR CODE HERE:
#   Evaluate the testing error of the k-nearest neighbors classifier
#   for your optimal hyperparameters found by 5-fold cross-validation.
# ================================================================ #
dists_L2 = knn.compute_distances(X=X_test, norm = L1_norm)
y_pred = knn.predict_labels(dists_L2, k=5)#/dists_L2.shape[0]
accuracy = np.mean(y_pred == y_test)
error = 1 - accuracy
pass

# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #

print('Error rate achieved: {}'.format(error))

