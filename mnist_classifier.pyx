# mnist_classifier.pyx
# cython: language_level=3
# cython: profile=True

import numpy as np
cimport numpy as np
from time import time

ctypedef np.float32_t dtype_float_t
ctypedef np.int32_t dtype_int_t

total_forward_prop_time = 0.0
total_backward_prop_time = 0.0

def train(np.ndarray[dtype_float_t, ndim=2] X_train, np.ndarray[dtype_int_t, ndim=1] Y_train, float alpha, int iterations):
    global total_forward_prop_time, total_backward_prop_time

    cdef int m_train = X_train.shape[1]

    cdef np.ndarray[dtype_float_t, ndim=2] W1, W2
    cdef np.ndarray[dtype_float_t, ndim=1] b1, b2

    def init_params():
        cdef np.ndarray[dtype_float_t, ndim=2] W1 = np.random.rand(10, 784).astype(np.float32) - 0.5
        cdef np.ndarray[dtype_float_t, ndim=1] b1 = (np.random.rand(10) - 0.5).astype(np.float32)
        cdef np.ndarray[dtype_float_t, ndim=2] W2 = np.random.rand(10, 10).astype(np.float32) - 0.5
        cdef np.ndarray[dtype_float_t, ndim=1] b2 = (np.random.rand(10) - 0.5).astype(np.float32)
        return W1, b1, W2, b2

    def ReLU(np.ndarray[dtype_float_t, ndim=2] Z):
        return np.maximum(Z, 0)

    def softmax(np.ndarray[dtype_float_t, ndim=2] Z):
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        return A

    def forward_prop(np.ndarray[dtype_float_t, ndim=2] W1, np.ndarray[dtype_float_t, ndim=1] b1,
                     np.ndarray[dtype_float_t, ndim=2] W2, np.ndarray[dtype_float_t, ndim=1] b2,
                     np.ndarray[dtype_float_t, ndim=2] X):
        global total_forward_prop_time
        start_time = time()  # INICIO DE PROFILING
        cdef np.ndarray[dtype_float_t, ndim=2] Z1 = W1.dot(X) + b1[:, np.newaxis]
        cdef np.ndarray[dtype_float_t, ndim=2] A1 = ReLU(Z1)
        cdef np.ndarray[dtype_float_t, ndim=2] Z2 = W2.dot(A1) + b2[:, np.newaxis]
        cdef np.ndarray[dtype_float_t, ndim=2] A2 = softmax(Z2)
        end_time = time()  # FINAL DEL PROFILING
        total_forward_prop_time += (end_time - start_time) #  TIEMPO ACUMULADO
        return Z1, A1, Z2, A2

    def ReLU_deriv(np.ndarray[dtype_float_t, ndim=2] Z):
        return Z > 0

    def one_hot(np.ndarray[dtype_int_t, ndim=1] Y):
        cdef int max_val = Y.max()
        cdef np.ndarray[dtype_float_t, ndim=2] one_hot_Y = np.zeros((Y.size, max_val + 1), dtype=np.float32)
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def backward_prop(np.ndarray[dtype_float_t, ndim=2] Z1, np.ndarray[dtype_float_t, ndim=2] A1,
                      np.ndarray[dtype_float_t, ndim=2] Z2, np.ndarray[dtype_float_t, ndim=2] A2,
                      np.ndarray[dtype_float_t, ndim=2] W1, np.ndarray[dtype_float_t, ndim=2] W2,
                      np.ndarray[dtype_float_t, ndim=2] X, np.ndarray[dtype_int_t, ndim=1] Y):
        global total_backward_prop_time
        start_time = time()  #INICIO DE PROFILING
        cdef np.ndarray[dtype_float_t, ndim=2] one_hot_Y = one_hot(Y)
        cdef np.ndarray[dtype_float_t, ndim=2] dZ2 = A2 - one_hot_Y
        cdef np.ndarray[dtype_float_t, ndim=2] dW2 = 1 / m_train * dZ2.dot(A1.T)
        cdef np.ndarray[dtype_float_t, ndim=1] db2 = 1 / m_train * np.sum(dZ2, axis=1)
        cdef np.ndarray[dtype_float_t, ndim=2] dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
        cdef np.ndarray[dtype_float_t, ndim=2] dW1 = 1 / m_train * dZ1.dot(X.T)
        cdef np.ndarray[dtype_float_t, ndim=1] db1 = 1 / m_train * np.sum(dZ1, axis=1)
        end_time = time()  #FINAL DE PROFILING
        total_backward_prop_time += (end_time - start_time)
        return dW1, db1, dW2, db2

    def update_params(np.ndarray[dtype_float_t, ndim=2] W1, np.ndarray[dtype_float_t, ndim=1] b1,
                      np.ndarray[dtype_float_t, ndim=2] W2, np.ndarray[dtype_float_t, ndim=1] b2,
                      np.ndarray[dtype_float_t, ndim=2] dW1, np.ndarray[dtype_float_t, ndim=1] db1,
                      np.ndarray[dtype_float_t, ndim=2] dW2, np.ndarray[dtype_float_t, ndim=1] db2,
                      float alpha):
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        return W1, b1, W2, b2

    def get_predictions(np.ndarray[dtype_float_t, ndim=2] A2):
        return np.argmax(A2, axis=0).astype(np.int32)

    def get_accuracy(np.ndarray[dtype_int_t, ndim=1] predictions, np.ndarray[dtype_int_t, ndim=1] Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(np.ndarray[dtype_float_t, ndim=2] X, np.ndarray[dtype_int_t, ndim=1] Y,
                         float alpha, int iterations):
        global total_forward_prop_time, total_backward_prop_time
        total_forward_prop_time = 0.0
        total_backward_prop_time = 0.0

        cdef np.ndarray[dtype_float_t, ndim=2] W1, W2
        cdef np.ndarray[dtype_float_t, ndim=1] b1, b2
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 100 == 0 or i == iterations - 1:
                predictions = get_predictions(A2)
                accuracy = get_accuracy(predictions, Y)
                print(f"Iteration {i}/{iterations} - Accuracy: {accuracy:.4f}")
        # PRINT DEL TOTAL
        print(f"Tiempo total de la función forward_prop time: {total_forward_prop_time:.4f} segundos")
        print(f"Tiempo total de la funcion backward_prop time: {total_backward_prop_time:.4f} segundos")
        return W1, b1, W2, b2

    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, iterations)
    return W1, b1, W2, b2
