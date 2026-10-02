import numpy as np

from src.subtareasPerceptron import propagar_hacia_adelante, calcular_delta_salida, calcular_delta_oculto, actualizar_pesos_salida, actualizar_pesos_ocultos, calcular_error





# diseño: unica capa oculta = unidades que quieras, unica salida = unidades que quieras, funcion de activacion = la que quieras, derivada de la funcion de activacion = la que quieras
def perceptron_multicapa(x, y, cota, n, num_oculta, num_salida, funcion_activacion_salida, funcion_activacion_oculta, derivada_activacion_salida, derivada_activacion_oculta):
    # inicializamos parametros iniciales para el algoritmo del multicapa
    # w_ocultos = np.random.rand(num_oculta, x.shape[1] + 1)  # pesos de la capa oculta (incluyendo bias)
    # w_salida = np.random.rand(num_salida, num_oculta + 1)  # pesos de la capa de salida (incluyendo bias)
    w_ocultos = np.random.uniform(-1, 1, (num_oculta, x.shape[1] + 1))  # pesos de la capa oculta (incluyendo bias)
    w_salida = np.random.uniform(-1, 1, (num_salida, num_oculta + 1))  # pesos de la capa de salida (incluyendo bias)
    # w_ocultos = np.zeros((num_oculta, x.shape[1] + 1))  # pesos de la capa oculta (incluyendo bias)
    # w_salida = np.zeros((num_salida, num_oculta + 1))  # pesos de la capa de salida (incluyendo bias)

    i = 0  # contador de iteraciones
    error = 1  # error inicializado en un valor alto
    error_min = float('inf')  # error minimo inicializado en un valor alto
    w_min = [w_ocultos.copy(), w_salida.copy()]  # pesos minimos inicializados en los pesos iniciales

    print("Empieza el entrenamiento del perceptron multicapa....................")

    while error > 0 and i < cota:
        i_x = np.random.randint(0, x.shape[0])  # seleccionamos un indice aleatorio de la muestra de datos de entrada
        # xi = np.append(x[i_x], 1)  # agregamos el bias a la entrada seleccionada, lo hacemos mejor dentro de la funcion propagar_hacia_adelante para que sea mas limpio y reutilizable

        # Propagacion hacia adelante
        h_oculto, o_oculto = propagar_hacia_adelante(x[i_x], w_ocultos, funcion_activacion_oculta)  # salida de la capa oculta
        print(f"\nSalida Final ponderada de la capa oculta: {h_oculto}")
        print(f"Salida Final de la capa oculta: {o_oculto}")

        h_salida, o_salida = propagar_hacia_adelante(o_oculto, w_salida, funcion_activacion_salida)  # salida de la capa de salida
        print(f"\nSalida Final ponderada de la capa de salida: {h_salida}")
        print(f"Salida Final de la capa de salida: {o_salida}")

    #     # Retropropagacion del error y actualizacion de pesos
    #     # print(f"\nRetropropagacion del error calculando deltas de salidas y ocultas ....................")
    #     delta_salida = calcular_delta_salida(o_salida, y[i_x], h_salida, derivada_activacion_salida)
    #     delta_oculto = calcular_delta_oculto(delta_salida, w_salida, h_oculto, derivada_activacion_oculta)

    #     # print(f"\nResultados de los deltas calculados....................")
    #     # print(f"Delta de la capa de salida: {delta_salida}")
    #     # print(f"Delta de la capa oculta: {delta_oculto}")

    #     # print(f"\nActualizacion de pesos de la capa de salida y oculta ....................")
    #     # print("para pesos de salida: ")
    #     # print(f"Parametros: w_salida: {w_salida}, delta_salida: {delta_salida}, o_oculto: {o_oculto}, n: {n}")

    #     # print("para pesos de la capa oculta: ")
    #     # print(f"Parametros: w_ocultos: {w_ocultos}, delta_oculto: {delta_oculto}, x[i_x]: {x[i_x]}, n: {n}")

    #     # Actualizamos los pesos de la capa de salida
    #     w_salida = actualizar_pesos_salida(w_salida, delta_salida, o_oculto, n)  # nuevo_w = n * delta_salida * o_oculto + w_salida
    #     # print(f"\nResultado pesos de la capa de salida actualizados: {w_salida}")

    #     # Actualizamos los pesos de la capa oculta
    #     w_ocultos = actualizar_pesos_ocultos(w_ocultos, delta_oculto, x[i_x], n)  # agregamos el bias a la entrada
    #     # print(f"\nResultado pesos de la capa oculta actualizados: {w_ocultos}")

    #     # Calculamos el error de los pesos encontrados en la iteracion actual
    #     error = calcular_error(x, y, w_salida, w_ocultos, funcion_activacion_oculta, funcion_activacion_salida)

    #     print(f"\nError total: {error}, error minimo: {error_min}, iteracion: {i} / {cota}, entrada: {x[i_x]}, salida final: {o_salida}, salida esperada: {y[i_x]} , w_ocultos: {w_ocultos}, w_salida: {w_salida}")

    #     if error < error_min:

    #         error_min = error
    #         w_min = [w_ocultos.copy(), w_salida.copy()]

    #         # print("-----------------------------------------------")
    #         # print(f"\nNuevo error minimo encontrado: {error_min}, error: {error}, pesos encontrados: {w_min}")
    #         # print("-----------------------------------------------")

    #     i += 1  # end

    # return i, w_min, error_min  # retornamos la cantidad de iteraciones realizadas, los pesos encontrados y el error minimo encontrado

    return





























