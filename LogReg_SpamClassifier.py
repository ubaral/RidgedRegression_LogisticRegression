import numpy as np
import scipy.io
import sklearn.preprocessing
import math
import LogisticRegressionUtils as logregUtils
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=500)  # better printing for matrices


def stochastic_kernel_grad_descent(step_size_init, train_data, train_labels, max_iter, decaying_learning_rate):
    n = np.shape(train_data)[0]
    kernel_mat = np.matrix(np.empty((n, n)))

    def k(x, z):
        p = 0.15  # hyper-parameter tuned with cross validation
        return ((x * z.getT())[0, 0] + p) ** 2

    for i in range(n):
        for j in range(n):
            kernel_mat[i, j] = k(np.matrix(train_data[i]), np.matrix(train_data[j]))
    risk_amt = []
    step_size = step_size_init  # a.k.a. learning rate, or epsilon
    a_vec_curr = np.matrix(np.ones((n, 1)))
    a_vec_next = logregUtils.update_weight_vector_kernel_ridged(a_vec_curr, kernel_mat, train_labels, step_size, 0,
                                                                decaying_learning_rate)  # update once
    iteration = 0

    for _ in range(max_iter):
        print("current risk on iteration {0} is {1}"
              .format(iteration, logregUtils.kernel_risk_func(a_vec_curr, kernel_mat, train_labels)))
        risk_amt.append(logregUtils.kernel_risk_func(a_vec_curr, kernel_mat, train_labels))
        a_vec_curr = a_vec_next
        a_vec_next = logregUtils.update_weight_vector_stochastic(a_vec_curr, kernel_mat, train_labels, step_size,
                                                                 iteration, decaying_learning_rate)
        iteration += 1
    return risk_amt


def stochastic_grad_descent(step_size_init, w_init, train_data, train_labels, max_iter, decaying_learning_rate):
    risk_amt = []
    step_size = step_size_init  # a.k.a. learning rate, or epsilon
    w_curr = w_init
    w_next = logregUtils.update_weight_vector_stochastic(w_curr, train_data, train_labels, step_size, 0,
                                                         decaying_learning_rate)  # update once
    iteration = 0
    for _ in range(max_iter):
        # print("current risk on iteration {0} is {1}"
        #       .format(iteration, logregUtils.risk_func(w_curr, training_data, training_labels)))
        risk_amt.append(logregUtils.risk_func(w_curr, training_data, training_labels))
        w_curr = w_next
        w_next = logregUtils.update_weight_vector_stochastic(w_curr, training_data, training_labels, step_size,
                                                             iteration, decaying_learning_rate)
        iteration += 1
    return w_next


def batch_grad_descent(step_size, w_init, train_data, train_labels, max_iter):
    # Use batch gradient descent to implement logistic regression
    step_size = step_size  # a.k.a. learning rate, or epsilon
    w_curr = w_init
    w_next = logregUtils.update_weight_vector_batch(w_curr, step_size, train_data, train_labels)  # update once
    iteration = 0
    for _ in range(max_iter):
        # print("current risk on iteration {0} is {1}"
        #       .format(iteration, logregUtils.risk_func(w_curr, training_data, training_labels)))
        w_curr = w_next
        w_next = logregUtils.update_weight_vector_batch(w_curr, step_size, training_data, training_labels)
        iteration += 1
    return w_next


# Load in data from provided spam dataset
mat = scipy.io.loadmat("spam_dataset/spam_data.mat")
training_data = np.matrix(mat["training_data"])
training_labels = np.matrix(mat["training_labels"]).getT()
test_data = np.matrix(mat["test_data"])
number_of_samples = np.shape(training_data)[0]

# # For the alpha offset term, we add a column of one to the training data, so the last weight represents alpha
# b = np.ones((number_of_samples, np.shape(training_data)[1] + 1))
# b[:, : -1] = training_data
# training_data = b

# (i) Standardize each column to have mean 0 and unit variance.
normalized_centered_train_data = np.matrix(sklearn.preprocessing.scale(training_data))
good_step = 3e-4

# (ii) Transform the features using X ij ← log(X ij +0.1), where the X ij ’s are the entries of the design matrix.
log_transformed_train_data = np.matrix(np.vectorize(lambda a: math.log(a + 0.1))(training_data))

# (iii) Binarize the features using X ij ← I(X ij > 0). I denotes an indicator variable.
binarized_train_data = np.matrix(sklearn.preprocessing.binarize(training_data))

# run gradient descent with specified pre-processing and store for plotting
maxIterations = 100
print(maxIterations)
iter_vals = [i for i in range(maxIterations)]
pre_proc = "(i)"
# stochastic works well with learning rate = 1
w_init = np.matrix(np.ones((np.shape(training_data)[1], 1)))
final_w = batch_grad_descent(3e-4, w_init, normalized_centered_train_data, training_labels, maxIterations)

f = open("output.csv", 'w')
f.write("Id,Category\n")
predicted = test_data * final_w
predicted = np.vectorize(lambda x: logregUtils.sigmoid_func(x))(predicted)
predicted2 = np.matrix(sklearn.preprocessing.binarize(predicted, .5))
print(predicted2)
for i in range(np.shape(test_data)[0]):
    f.write(str(i + 1) + "," + str(int(predicted2[i, 0])) + "\n")
print("DONE!")

# print("For Stochastic G.D. #{2} MIN risk was: {0}, MAX risk was: {1}".format(min(risk_vals), max(risk_vals), pre_proc))
# fig = plt.gcf()
# plt.title("Batch GD".format(pre_proc))
# plt.ylabel('Risk')
# plt.xlabel('Number of Iterations')
# fig.canvas.set_window_title('Batch GD'.format(pre_proc))
# plt.plot(iter_vals, risk_vals, '-bo')
#
# pre_proc = "(ii)"
# # stochastic works well with learning rate = 1
# w_init = np.matrix(np.ones((np.shape(training_data)[1], 1)))
# risk_vals = stochastic_grad_descent(1e-1, w_init, log_transformed_train_data, training_labels, maxIterations, True)
#
# print("For Stochastic G.D. #{2} MIN risk was: {0}, MAX risk was: {1}".format(min(risk_vals), max(risk_vals), pre_proc))
# plt.plot(iter_vals, risk_vals, '-ro')
#
# pre_proc = "(iii)"
# # stochastic works well with learning rate = 1
# w_init = np.matrix(np.ones((np.shape(training_data)[1], 1)))
# risk_vals = stochastic_grad_descent(1e-1, w_init, binarized_train_data, training_labels, maxIterations, True)
#
# print(
#     "For Stochastic G.D. #{2} MIN risk was: {0}, MAX risk was: {1}".format(min(risk_vals), max(risk_vals), pre_proc), )
# plt.plot(iter_vals, risk_vals, '-go')
#
# plt.show()
