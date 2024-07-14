# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:38:49 2024

@author: joaqu
"""

import numpy as np
import pandas as pd
import datetime
import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from matplotlib import pyplot as plt
import cProfile
import pstats
import io

# Función para el profiling
def profile_code():
    data = pd.read_csv('C:/Users/joaqu/Desktop/Academia/MAGISTER/Herramientas de computacion de altorendimiento/MNIST/train.csv')

    data = np.array(data)

    m, n = data.shape

    np.random.shuffle(data) 

    data_dev = data[0:1000].T

    _Y_dev = data_dev[0]
    Y_dev = jnp.array(_Y_dev)
    _X_dev = data_dev[1:n]
    X_dev = jnp.array(_X_dev)

    X_dev = X_dev / 255.

    data_train = data[1000:m].T

    Y_train = data_train[0]
    X_train = data_train[1:n]

    X_train = X_train / 255.

    _, m_train = X_train.shape

    Y_train

    def _init_params():
        key = jax.random.PRNGKey(0)
        W1 = jax.random.uniform(key, (10, 784), minval=-0.5, maxval=0.5)
        b1 = jax.random.uniform(key, (10, 1), minval=-0.5, maxval=0.5)
        W2 = jax.random.uniform(key, (10, 10), minval=-0.5, maxval=0.5)
        b2 = jax.random.uniform(key, (10, 1), minval=-0.5, maxval=0.5)
        return W1, b1, W2, b2

    def ReLU(Z):
        return jnp.maximum(Z, 0)

    def softmax(Z):
        A = jnp.exp(Z) / sum(jnp.exp(Z))
        return A
    
    def forward_prop(W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2
    
    def calentamiento ():

        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1)
        

    def ReLU_deriv(Z):
        return Z > 0

    def one_hot(Y):
        one_hot_Y = jnp.zeros((Y.size, Y.max() + 1))
        one_hot_Y = one_hot_Y.at[jnp.arange(Y.size), Y].set(1)
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * jnp.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * jnp.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1    
        W2 = W2 - alpha * dW2  
        b2 = b2 - alpha * db2    
        return W1, b1, W2, b2

    def get_predictions(A2):
        return jnp.argmax(A2, 0)

    def get_accuracy(predictions, Y):
        return jnp.sum(predictions == Y) / Y.size

    def gradient_descent(X, Y, alpha, iterations):
        init_params = jax.jit(_init_params)
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        return W1, b1, W2, b2


    calentamiento()



    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)


if __name__ == '__main__':

    profile_code()

