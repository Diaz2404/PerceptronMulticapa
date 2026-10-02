"""
2) Discriminar si un número es par, con entradas dadas por el conjunto de números
decimales del 0 al 9 (usar archivo TP2-ej3-mapa-de-pixeles-digitos-decimales.txt)
representados por imágenes de 5 x 7 pixeles.
Entrene con un subconjunto de los dígitos y utilice el resto para testear a la red. ¿Qué podría
decir acerca de la capacidad para generalizar de la red?

"""

import numpy as np

# entradas es un vector que guarda los 10 digitos en un vector unimensional de 35 elementos/pixeles (5x7)
entradas =np.array([
    # 0
    [0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,1,1,
     1,0,1,0,1,
     1,1,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 1
    [0,0,1,0,0,
     0,1,1,0,0,
     0,0,1,0,0,
     0,0,1,0,0,
     0,0,1,0,0,
     0,0,1,0,0,
     0,1,1,1,0],

    # 2
    [0,1,1,1,0,
     1,0,0,0,1,
     0,0,0,0,1,
     0,0,0,1,0,
     0,0,1,0,0,
     0,1,0,0,0,
     1,1,1,1,1],

    # 3
    [0,1,1,1,0,
     1,0,0,0,1,
     0,0,0,0,1,
     0,0,1,1,0,
     0,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 4
    [0,0,0,1,0,
     0,0,1,1,0,
     0,1,0,1,0,
     1,0,0,1,0,
     1,1,1,1,1,
     0,0,0,1,0,
     0,0,0,1,0],

    # 5
    [1,1,1,1,1,
     1,0,0,0,0,
     1,1,1,1,0,
     0,0,0,0,1,
     0,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 6
    [0,0,1,1,0,
     0,1,0,0,0,
     1,0,0,0,0,
     1,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 7
    [1,1,1,1,1,
     0,0,0,0,1,
     0,0,0,1,0,
     0,0,1,0,0,
     0,1,0,0,0,
     0,1,0,0,0,
     0,1,0,0,0],

    # 8
    [0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],

    # 9
    [0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,1,
     0,0,0,0,1,
     0,0,0,1,0,
     0,1,1,0,0]])



salidas_deseadas = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])  # salidas deseadas para cada digito (par = 1, impar = -1)


# entrenamos el modelo con los primeros 8 digitos (0-7) y testeamos con los ultimos 2 digitos (8-9)
x_train = entradas[:8]  # Entrenamiento con los digitos 0-
y_train = salidas_deseadas[:8]  # Salidas deseadas para el entrenamiento

x_test = entradas[8:]  # Testeamos con los digitos 8-9
y_test = salidas_deseadas[8:]  # Salidas deseadas para el test

from src.PerceptronMulticapa import perceptron_multicapa
from src.funcionActivacion import activacion_tangente, activacion_tangente_derivada, activacion_logistica, activacion_logistica_derivada, activacion_lineal, activacion_lineal_derivada
from src.subtareasPerceptron import predecir

i,w,error = perceptron_multicapa(x_train, y_train, cota=2000, n=0.1, num_oculta=32, num_salida=1,funcion_activacion_salida=activacion_tangente, funcion_activacion_oculta=activacion_tangente, derivada_activacion_salida=activacion_tangente_derivada, derivada_activacion_oculta=activacion_tangente_derivada)

print(f"\nEntrenamiento finalizado en {i} iteraciones con un error de {error}")

# predecimos con los datos de train 0-7
for i in range(len(x_train)):
    prediccion = predecir(x_train[i], w[0], w[1], activacion_tangente, activacion_tangente)
    print(f"Prediccion para el digito {i}: {prediccion}, salida esperada: {y_train[i]}")

# predecimos con los datos de testeo 8-9
for i in range(len(x_test)):
    prediccion = predecir(x_test[i], w[0], w[1], activacion_tangente, activacion_tangente)
    print(f"Prediccion para el digito {i+8}: {prediccion}, salida esperada: {y_test[i]}")


# Conclusion sobre la capacidad de generalización de la red:
# La red entrenada con los dígitos del 0 al 7 logra predecir correctamente si un número es par o impar para esos dígitos.
# Sin embargo, al probar con los dígitos 8 y 9, que no fueron parte del conjunto de entrenamiento, la red puede no generalizar correctamente.
# Esto indica que la capacidad de generalización de la red es limitada y depende del conjunto de entrenamiento utilizado. Para mejorar la generalización, sería recomendable entrenar la red con un conjunto más amplio de dígitos o utilizar técnicas de regularización.



