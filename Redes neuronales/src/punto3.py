"""
3) Construba un perceptron multicapa con la misma entrada usada en 2) pero con 10
unidades de salidas de modo que cada salida represente a un dígito (por ejemplo, si se le
presenta a la red la imagen del 7, el estado de la unidad de salida número dígito 7 esté en 1
y las restantes en 0).
Una vez que la red haya aprendido, use patrones de entrada correspondientes a los dígitos
de entrenamiento pero con sus píxeles afectados por ruido (por ejemplo, con probabilidad
0,02, intercambie el valor de los bits en la imagen del dígito) y evalúe los resultados.
"""

from src.PerceptronMulticapa import perceptron_multicapa
from src.funcionActivacion import activacion_tangente, activacion_tangente_derivada, activacion_logistica, activacion_logistica_derivada
from src.subtareasPerceptron import predecir
import numpy as np

# Entradas: 10 dígitos representados como imágenes de 5x7 píxeles (35 elementos)
entradas = np.array([
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
     0,1,1,0,0]
])




# EVALUAR SIN RUIDO Y SIN TESTEO DE VALIDACION
salida_one_hot = np.eye(10) # crea matriz identidad de 10x10, que representa la codificación one-hot para los dígitos del 0 al 9
x_train = entradas
y_train = salida_one_hot

i, w, error = perceptron_multicapa(x_train , y_train, cota=10000, n=0.01,num_oculta=32, num_salida=10, funcion_activacion_salida=activacion_logistica, funcion_activacion_oculta=activacion_logistica, derivada_activacion_salida=activacion_logistica_derivada, derivada_activacion_oculta=activacion_logistica_derivada)

print(f"\nEntrenamiento finalizado en {i} iteraciones con error: {error}")

print("\n\nEvaluando sin ruido en los datos de entrenamiento:\n")
for i in range(len(x_train)):
    prediccion = predecir(x_train[i], w[0], w[1], activacion_logistica, activacion_logistica)
    print(f"Dígito {i}: Predicción Sin Ruido={np.argmax(prediccion)}, \nSalida= {prediccion} \n{y_train[i]} \n")


# EVALUAR CON RUIDO Y SIN TESTEO DE VALIDACION
def agregar_ruido_al_vector(vector, probabilidad):
    vector_ruidoso = vector.copy()
    for i in range(len(vector)):
        if np.random.rand() < probabilidad:
            vector_ruidoso[i] = 1 - vector_ruidoso[i]  # Cambiamos el bit (0 a 1 o 1 a 0)
    return vector_ruidoso   

print("\n\nEvaluando con ruido en los datos de entrenamiento (probabilidad de ruido = 0.02):\n")

# Predecimos con ruido en los datos de entrenamiento
for i in range(len(x_train)):
    x_train_ruidoso = agregar_ruido_al_vector(x_train[i], probabilidad=0.02)
    prediccion = predecir(x_train_ruidoso, w[0], w[1], activacion_logistica, activacion_logistica)
    print(f"Dígito {i}: Predicción Con Ruido={np.argmax(prediccion)}, \nSalida= {prediccion} \n{y_train[i]} \n")











