import numpy as np
import pandas as pd
import mnist_classifier
import cProfile

def main():
    # Cargar datos usando pandas para manejar el encabezado
    data = pd.read_csv('C:/Users/joaqu/Desktop/Academia/MAGISTER/Herramientas de computacion de altorendimiento/MNIST/train.csv')
    
    # Convertir el DataFrame a un array de Numpy
    data = data.to_numpy()
    
    # Se consiguen las dimensiones del array.
    m, n = data.shape
    
    # Barajar los datos para asegurarse que los datos sean válidos para el entrenamiento del modelo
    np.random.shuffle(data)
    
    # Se toman las primeras 1000 filas del array barajado y se le aplica la trasposición T
    data_dev = data[0:1000].T
    
    # Se separa las etiquetas y características del conjunto de desarrollo
    Y_dev = data_dev[0].astype(np.int32)
    X_dev = data_dev[1:n].astype(np.float32)
    
    # Normaliza la característica dividiéndola en 255
    X_dev = X_dev / 255.
    
    # Se toman las filas desde el 1000 hasta el resto de las filas y se aplica la trasposición T
    data_train = data[1000:m].T
    
    # Y_train se establece como la primera fila del data_train que contiene las etiquetas del conjunto de entrenamiento
    Y_train = data_train[0].astype(np.int32)
    X_train = data_train[1:n].astype(np.float32)
    
    # Normaliza la característica dividiéndola en 255
    X_train = X_train / 255.
    
    # Profiling de la función train
    profiler = cProfile.Profile()
    profiler.enable()
    
    mnist_classifier.train(X_train, Y_train, 0.10, 500)
    
    profiler.disable()
    profiler.dump_stats('mnist_profiling.prof')

if __name__ == '__main__':
    main()
    print("listo ;D")