import numpy as np


"""
CONSIGNA:

Implemente un perceptron multicapa y utilíselo para aprender los siguientes problemas:
1) Función lógica O exclusivo con entradas
x = {{-1, 1}, {1, -1}, {-1, -1}, {1, 1}},
y salida esperada y = {1, 1, -1, -1}
"""


from src.funcionActivacion import activacion_lineal, activacion_lineal_derivada, activacion_logistica, activacion_logistica_derivada, activacion_tangente, activacion_tangente_derivada
from src.subtareasPerceptron import predecir
from src.PerceptronMulticapa import perceptron_multicapa


x_xor = np.array([
[-1,  1],
[ 1, -1],
[-1, -1],
[ 1,  1]
])

y_xor = np.array([
    1,
    1,
    -1,
    -1
])



i_xor,w_min_xor, error_min_xor = perceptron_multicapa(x_xor, y_xor, cota=2000, n=0.01, num_oculta=4, num_salida=1, funcion_activacion_salida=activacion_tangente, funcion_activacion_oculta=activacion_tangente, derivada_activacion_salida=activacion_tangente_derivada, derivada_activacion_oculta=activacion_tangente_derivada)


print(f"\nEntrenamiento del perceptron multicapa finalizado....................")
print(f"Iteraciones realizadas: {i_xor}")
print(f"Pesos encontrados: \n{w_min_xor}")
print(f"Error minimo encontrado: {error_min_xor}")

print(f"\nPredicciones del perceptron multicapa....................")
for i in range(x_xor.shape[0]):
    prediccion = predecir(x_xor[i], w_min_xor[0], w_min_xor[1], activacion_tangente, activacion_tangente)
    if prediccion >= 0:
        prediccion = 1
    elif prediccion <= -0:
        prediccion = -1
    print(f"Entrada: {x_xor[i]}, Salida esperada: {y_xor[i]}, Salida predicha: {prediccion}")






