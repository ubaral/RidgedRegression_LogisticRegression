import numpy as np
import scipy.io
import random

mat = scipy.io.loadmat("housing_dataset/housing_data.mat")
Xtrain = np.matrix(mat["Xtrain"])
Ytrain = np.matrix(mat["Ytrain"])
N = np.shape(Ytrain)[0]

print("Size of Xtrain" + str(np.shape(Xtrain)))
print("Size of Xtrain" + str(np.shape(Ytrain)))

# A = Xtrain.getT()*Xtrain+np.matrix(penalty_coeff*(np.identity(np.shape(Xtrain)[1])))
# b = Xtrain.getT()*Ytrain
# alpha_hat = (1/N)*np.sum(Ytrain)
# w_hat = np.linalg.solve(A,b)

#10-fold cross validation to tune hyperparamater lambda
k = 10
partitionSize = N // k
indexSet = set([i for i in range(N)])
# each k integer value 0,1,2,...,9 will map to the kth partitioning of the 10000 images.
# Each partition will be a list of a list of images and a list of corresponding labels,
# for training/validation purposes.
partitionDict = dict()
for i in range(k):
    partitionDict[i] = [np.matrix(np.empty((0, 8), Xtrain[0,0].dtype)), np.matrix(np.empty((0, 1), Ytrain[0,0].dtype))]

# Here we select 10000 random images from the full set of 60,000 images we have.
# In the loop we will mod by k and send that image to the appropriate "bucket" partition that it belongs to.
for _ in range(N):
    matrixIndex = _ // k
    partitionKey = _ % k
    i = random.sample(indexSet, 1)[0]
    indexSet.remove(i)
    partitionDict[partitionKey][0] = np.append(partitionDict[partitionKey][0], Xtrain[i], axis = 0)
    partitionDict[partitionKey][1] = np.append(partitionDict[partitionKey][1], Ytrain[i], axis = 0)

#Outer loop represents partition to validate on, and inner loop will combine all other partitions, and use it as training data for the ridge regression.
Avg_RSS_Dict = dict();
for penalty_coeff in range(20,200):
    Avg_RSS = 0;
    for key in partitionDict:
        validationPartition = partitionDict[key]
        validation_X = validationPartition[0]
        validation_y = validationPartition[1]

        cv_A = np.matrix(np.empty((0, 8), Xtrain[0,0].dtype))
        cv_b = np.matrix(np.empty((0, 1), Ytrain[0,0].dtype))

        for otherKey in partitionDict:
            if otherKey != key:
                trainingPartition = partitionDict[otherKey]
                cv_A = np.append(cv_A, trainingPartition[0], axis=0)
                cv_b = np.append(cv_b, trainingPartition[1], axis=0)

        A = cv_A.getT()*cv_A+np.matrix(penalty_coeff*(np.identity(np.shape(cv_A)[1])))
        b = cv_A.getT()*cv_b
        alpha_hat = (1/N)*np.sum(cv_b)
        w_hat = np.linalg.solve(A,b)

        RSS = 0
        for i in range(np.shape(validation_X)[0]):
            X_i = validation_X[i]
            RSS += ((X_i*w_hat)[0,0] + alpha_hat - validation_y[i,0])**2
        Avg_RSS += RSS

    Avg_RSS = Avg_RSS / k
    print("Average RSS for 10-fold validation with lamda=" + str(penalty_coeff) + " is " + str(Avg_RSS))
    Avg_RSS_Dict[penalty_coeff] = Avg_RSS

min(Avg_RSS_Dict, key=Avg_RSS_Dict.get)