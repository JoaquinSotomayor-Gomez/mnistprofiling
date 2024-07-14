import numpy as np
import pandas as pd
import datetime
import cProfile
import pstats

def main():
    data = pd.read_csv('C:/Users/joaqu/Desktop/Academia/MAGISTER/Herramientas de computacion de altorendimiento/MNIST/train.csv')
    
    # Transformación de la variable data a un array de Numpy
    data = np.array(data)
    
    # Se consiguen las dimensiones del array.
    m, n = data.shape
    
    # Barajar los datos para asegurarse que los datos sean válidos para el entrenamiento del modelo
    np.random.shuffle(data)
    
    # Se toman las primeras 1000 filas del array barajado y se le aplica la trasposición T
    data_dev = data[0:1000].T
    
    # Se separa las etiquetas y características del conjunto de desarrollo
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    
    # Normaliza la característica dividiéndola en 255
    X_dev = X_dev / 255.
    
    # Se toman las filas desde el 1000 hasta el resto de las filas y se aplica la trasposición T
    data_train = data[1000:m].T
    
    # Y_train se establece como la primera fila del data_train que contiene las etiquetas del conjunto de entrenamiento
    Y_train = data_train[0]
    X_train = data_train[1:n]
    
    # Normaliza la característica dividiéndola en 255
    X_train = X_train / 255.
    
    # Obtiene el número de columnas (ejemplos) en X_train y lo guarda en m_train
    _, m_train = X_train.shape
    
    def init_params():
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2
    
    def ReLU(Z):
        return np.maximum(Z, 0)
    
    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    def forward_prop(W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2
    
    def ReLU_deriv(Z):
        return Z > 0
    
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2
    
    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2
    
    def get_predictions(A2):
        return np.argmax(A2, 0)
    
    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    def gradient_descent(X, Y, alpha, iterations):
        W1, b1, W2, b2 = init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            #if i % 10 == 0:
                #print("Iteration: ", i)
                #predictions = get_predictions(A2)
                #print(get_accuracy(predictions, Y))
        return W1, b1, W2, b2
    
    start = datetime.datetime.now()
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
    stop = datetime.datetime.now()
    duration = stop - start
    #print(duration)

if __name__ == '__main__':
   # profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats()