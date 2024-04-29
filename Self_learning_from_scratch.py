import numpy as np
import math
import random


X = np.array(
    [[4, 2, -1], [0.01, -1, 3.5], [0.01, 2, 0.01], [-1, 2.5, -2], [-1.5, 2, 1.5]]
)

T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


S = 3
K = 5


def weights_initialization(S, K):
    weights = np.random.uniform(low=-0.1, high=0.1, size=(K, S))
    return weights


def random_column(array):
    num_columns = array.shape[1]
    random_index = np.random.choice(num_columns)
    return array[:, random_index], random_index


# def learning():
#     learn_rate = 0.1
#     # for i in range(0, 10):
#     specific_column, index = random_column(X)
#     y1 = np.dot(weights_transposed, specific_column)
#     matrix_result = T[index] - y1
#     X_column = X[:, index]
#     matrix_result_transposed = matrix_result.transpose()
#     dw = learn_rate*np.dot(X_column, matrix_result_transposed)
#     print(dw)


def learning():
    learn_rate = 0.1
    for i in range(0, 100):
        specific_column, index = random_column(X)
        y1 = np.dot(weights_transposed, specific_column)
        matrix_result = T[index] - y1
        X_column = X[:, index].reshape(-1, 1)  # Reshape to column vector
        matrix_result_transposed = matrix_result.reshape(1, -1)  # Reshape to row vector
        dw = learn_rate * np.dot(X_column, matrix_result_transposed)
        print("\n", dw)


weights = weights_initialization(S, K)
weights_transposed = weights.transpose()
U = np.dot(weights_transposed, X)
y = 1 / (1 + pow(math.e, -5 * U))


print("\nMatrix of weights example:\n\n", weights)
print("\n\nMatrix X: \n", X)
print("\n\nMatrix X: \n", U)
print("\n\nActivation function y: \n\n", y)


learning()
