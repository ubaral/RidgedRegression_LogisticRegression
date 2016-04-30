import numpy as np
import math


# our sigmoid function, which returns:
#
#                  1
# s(gamma) = --------------
#            (1+e^-(gamma))
#
# Also, this function gives us the Pr(Y|X), or at least our estimate of it
def sigmoid_func(gamma):
    return 1 / (1 + math.pow(math.e, -gamma))


# N represents the number of samples we have, labels = y, weights_vec = w,
# and design_matrix = X. Where w is a column vector of weights, X is the
# (n * d) design matri
def risk_func(weights_vec, design_matrix, labels):
    risk = 0
    for i in range(np.shape(design_matrix)[0]):
        risk += labels[i, 0] * math.log(sigmoid_func((design_matrix[i] * weights_vec)[0, 0])) + (1 - labels[
            i, 0]) * math.log(max(1e-323, 1 - sigmoid_func((design_matrix[i] * weights_vec)[0, 0])))
    return -risk


# different risk function for kernel logistic regression
def kernel_risk_func(a_vec, kernel_mat, labels):
    risk = 0
    for i in range(np.shape(kernel_mat)[0]):
        risk += labels[i, 0] * math.log(sigmoid_func(kernel_mat[i] * a_vec)) + (1 - labels[
            i, 0]) * math.log(max(1e-323, 1 - sigmoid_func(kernel_mat[i] * a_vec)))
    return -risk


def compute_mu(weights, design_mat):
    mu = np.matrix(np.empty((np.shape(design_mat)[0], 1)))
    for i in range(np.shape(design_mat)[0]):
        mu[i, 0] = sigmoid_func((design_mat[i] * weights)[0, 0])
    return mu


def update_weight_vector_batch(w_i, learning_rate, design_mat, labels):
    summation = sum([(labels[i, 0] - sigmoid_func(design_mat[i] * w_i)) * (np.matrix(design_mat[i]).getT()) for i in
                     range(np.shape(design_mat)[0])])
    return w_i + learning_rate * summation


def update_weight_vector_stochastic(w_i, design_mat, labels, epsilon_0, t, decays):
    from random import randrange
    rand_index = randrange(0, np.shape(design_mat)[0])
    rand_samp = design_mat[rand_index]
    summation = (labels[rand_index, 0] - sigmoid_func(rand_samp * w_i)) * (np.matrix(rand_samp).getT())
    if decays:
        epsilon_t = epsilon_0 * (1 / (1 + t))
        return w_i + epsilon_t * summation
    else:
        return w_i + epsilon_0 * summation


def update_weight_vector_kernel_ridged(a_vec, kernel_mat, labels, epsilon_0, t, decays):
    lambd = 1e-3  # hyper-parameter to tune
    from random import randrange
    rand_index = randrange(0, np.shape(a_vec)[0])
    summation = (labels[rand_index, 0] - sigmoid_func(kernel_mat[rand_index] * a_vec))

    if decays:
        epsilon_t = epsilon_0 * (1 / (1 + t))
        a_vec[rand_index] = epsilon_t * summation - epsilon_t * lambd * a_vec[rand_index]
        for h in range(np.shape(a_vec)[0]):
            if h != rand_index:
                a_vec[h] = a_vec[h] - epsilon_t * lambd * a_vec[h]

    else:
        a_vec[rand_index] = epsilon_0 * summation - epsilon_0 * lambd * a_vec[rand_index]
        for h in range(np.shape(a_vec)[0]):
            if h != rand_index:
                a_vec[h] = a_vec[h] - epsilon_0 * lambd * a_vec[h]

    return a_vec;
