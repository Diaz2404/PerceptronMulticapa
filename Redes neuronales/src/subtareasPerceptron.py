import numpy as np



"""
i = índice de la neurona en la capa de salida
j = índice de la neurona en la capa oculta

Formula delta para la capa de salida:
delta_i = (y_pred_i - y_true_i) * derivada_funcion_activacion(h_i)

Formula delta para la capa oculta:
delta_j = Sumatoria(pesos_capa_ij * delta_i) * derivada_funcion_activacion(h_j)
"""

# Subtareas del perceptron multicapa:

def propagar_hacia_adelante(x, w, funcion_activacion):
    cant_neuronas = w.shape[0]  # Número de neuronas en la capa
    o = np.zeros(cant_neuronas)  # Inicializamos el vector de activación de la capa con ceros
    h = np.zeros(cant_neuronas)  # Inicializamos el vector de salida de la capa con ceros

    # print(f"\npesos de la capa: \n{w}")

    x_i = np.append(x, 1)  # Agregamos el bias a la entrada
    for i in range(cant_neuronas):
        h[i] = np.sum(w[i] * x_i)  # Producto punto entre los pesos de la neurona i y la entrada x_i (con bias)
        o[i] = funcion_activacion(h[i])  # Aplicamos la función de activación a la salida ponderada

        # print("\nCalculo ponderada de la capa.................... ")
        # print(f" Entrada: {x_i}, pesos: {w[i]}")
        # print(f"Producto punto: {h[i]}, salida activada: {o[i]}")


    return h, o  # h: np.array , o: np.array


def calcular_delta_salida(y_pred, y_true, h_salida, derivada_activacion):
    # print("Calculo delta de la capa de salida....................")
    # print(f"y_pred: {y_pred}, y_true: {y_true}, h_salida: {h_salida}")

    delta_salida = (y_true - y_pred) * derivada_activacion(h_salida)  # delta_i = (y_pred_i - y_true_i) * derivada_funcion_activacion(h_i)
    return delta_salida  # delta_i: np.array

def calcular_delta_oculto(delta_salida, w_salida, h_oculto, g_derivada):
    # print("Calculo delta de la capa oculta....................")
    # print(f"delta_salida: {delta_salida}, w_salida: {w_salida}, h_oculto: {h_oculto}")

    delta_oculto = np.zeros(h_oculto.shape[0])  # Inicializamos el vector de delta de la capa oculta con ceros
    w_sin_bias = w_salida[:, :-1]  # Eliminamos la columna de bias de los pesos de la capa de salida
    w_transpuesta = w_sin_bias.T  # Transponemos la matriz de pesos de la capa de salida para poder multiplicar con el delta de la capa de salida

    # print(f"\nw_salida: {w_salida}, \nw_sin_bias: {w_sin_bias}, \nw_transpuesta: \n{w_transpuesta}")
    # print(f"\nvariable que contendra cada delta de la neurona de la capa oculta: {delta_oculto}")


    # print(f"\nW_transpuesta: {w_transpuesta} , delta_salida: {delta_salida}")
    # print(f"{w_transpuesta[0]} * {delta_salida} = {w_transpuesta[0] * delta_salida}")


    # for i in range(delta_oculto.shape[0]):
    #     print(f"\nIteracion {i+1} / {delta_oculto.shape[0]}: ")
    #     print(f"Pesos salida transpuesta de la neurona {i+1}: {w_transpuesta[i]} de \n{w_transpuesta}")
    #     print(f"Multiplicacion del error propagado: {w_transpuesta[i]} * {delta_salida} = {w_transpuesta[i] * delta_salida}")
    #     print(f"g_derivada(h_oculto[{i}]): {g_derivada(h_oculto[i])}, error propagado: {np.sum(w_transpuesta[i] * delta_salida)}")


    #     delta_oculto[i] = g_derivada(h_oculto[i]) * np.sum(w_transpuesta[i] * delta_salida)  # delta_j = Sumatoria(pesos_capa_ij * delta_i) * derivada_funcion_activacion(h_j)
    #     print(f"Resultado guardado en delta_oculto[{i}]: {delta_oculto[i]}")

    # print(f"\nDelta de la capa oculta calculado: {delta_oculto}")

    # print(f"\nVariables iterativos para calcular cada delta de la capa oculta: ")
    for j in range(delta_oculto.shape[0]):
        delta_oculto[j] = g_derivada(h_oculto[j]) * np.sum(w_transpuesta[j] * delta_salida)  # delta_j = Sumatoria(pesos_capa_ij * delta_i) * derivada_funcion_activacion(h_j)

        # print(f"\nIteracion {j+1} / {delta_oculto.shape[0]}: ")
        # print(f"Pesos salida transpuesta de la neurona {j+1}: {w_transpuesta[j]} de \n{w_transpuesta}")
        # print(f"Resultado de w_transpuesta * delta_salida: {w_transpuesta[j] * delta_salida}, su calculo es: transpuesta de pesos * delta de salida: {w_transpuesta[j]} * {delta_salida}")
    return delta_oculto  # delta_j: np.array


def actualizar_pesos_salida(w, delta, o_anterior, n):
    # print("\nActualizando pesos..............")
    # print("Parametros para la actualizacion de pesos: ")
    # print(f"  - w: {w}")
    # print(f"  - delta: {delta}")
    # print(f"  - entrada: {o_anterior}")
    # print(f"  - n: {n}")
    o_anterior_con_bias = np.append(o_anterior, 1)  # Agregamos el bias a la entrada anterior para que tenga la misma dimension que los pesos
    return w + n * delta * o_anterior_con_bias


def actualizar_pesos_ocultos(w, delta, x, n):
    # print("\nActualizando pesos ocultos..............")
    # print("Parametros para la actualizacion de pesos ocultos: ")
    # print(f"  - w oculto: {w}")
    # print(f"  - delta oculto: {delta}")
    # print(f"  - entrada: {x}")
    # print(f"  - n: {n}")

    x_con_bias = np.append(x, 1)  # Agregamos el bias a la entrada para que tenga la misma dimension que los pesos

    # cuando w es una matriz de pesos, cada fila es la cant de neuronas de la capa oculta y cada columna es la cant de entradas + 1 (bias)
    for i in range(w.shape[0]):
        w[i] += n * delta[i] * x_con_bias
        # print(f"Actualizando pesos de la neurona {i+1}: {w[i]} += {n} * {delta[i]} * {x_con_bias}")
    return w



def calcular_error(x, y, w_salida, w_oculto,funcion_activacion_oculta, funcion_activacion_salida):
    # calculamos el error como la suma de los errores de cada capa, para esto calculamos el error de la capa de salida y el error de la capa oculta
    # print("\nCalculando error....................")
    error_total = 0

    for i in range(y.shape[0]):
        _, o_oculto = propagar_hacia_adelante(x[i], w_oculto, funcion_activacion_oculta)
        _, o_salida = propagar_hacia_adelante(o_oculto, w_salida, funcion_activacion_salida)

        error_total += (y[i] - o_salida) ** 2

        # print(f"Iteracion {i+1} / {y.shape[0]}: ")
        # print(f"Entrada: {x[i]}, salida final: {o_salida}, salida esperada: {y[i]}, error: {(y[i] - o_salida) ** 2}")
    return error_total / 2



def predecir(x, w_ocultos, w_salida, funcion_activacion_oculta, funcion_activacion_salida):
    # Propagacion hacia adelante
    _, o_oculto = propagar_hacia_adelante(x, w_ocultos, funcion_activacion_oculta)  # salida de la capa oculta
    _, o_salida = propagar_hacia_adelante(o_oculto, w_salida, funcion_activacion_salida)  # salida de la capa de salida

    return o_salida  # retornamos la salida final del perceptron multicapa





