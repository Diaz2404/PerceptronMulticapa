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


# PASO 1: Crear salidas con codificación one-hot (10 salidas, una para cada dígito)
def crear_salidas_one_hot(cantidad_digitos=10):
    """Crea matriz de salidas con codificación one-hot"""
    return np.eye(cantidad_digitos)


# salidas_one_hot = crear_salidas_one_hot(10)
salida_one_hot = np.eye(10) # crea matriz identidad de 10x10, que representa la codificación one-hot para los dígitos del 0 al 9
x_train = entradas
y_train = salida_one_hot

# PASO 2: Función para agregar ruido a las imágenes
def agregar_ruido(imagen, probabilidad_ruido=0.02):
    """Agrega ruido a una imagen intercambiando bits con cierta probabilidad."""
    imagen_con_ruido = imagen.copy()
    for i in range(len(imagen_con_ruido)):
        if np.random.rand() < probabilidad_ruido:
            imagen_con_ruido[i] = 1 - imagen_con_ruido[i]
    return imagen_con_ruido



# PASO 4: Entrenar el perceptrón multicapa con 10 salidas
print("=" * 70)
print("ENTRENAMIENTO DEL PERCEPTRÓN MULTICAPA CON 10 SALIDAS (ONE-HOT)")
print("=" * 70)



i, w, error = perceptron_multicapa(x_train , y_train, cota=4, n=0.1,num_oculta=3, num_salida=10, funcion_activacion_salida=activacion_logistica, funcion_activacion_oculta=activacion_logistica, derivada_activacion_salida=activacion_logistica_derivada, derivada_activacion_oculta=activacion_logistica_derivada)



# # PASO 5: Evaluar predicciones sin ruido
# print("PREDICCIONES SIN RUIDO (DATOS DE ENTRENAMIENTO):")
# print("-" * 70)

# aciertos_sin_ruido = 0
# for digito_idx in range(len(x_train)):
#     prediccion = predecir(x_train[digito_idx], w[0], w[1], activacion_tangente, activacion_logistica)
#     digito_predicho = np.argmax(prediccion)
#     digito_esperado = digito_idx
    
#     es_correcto = digito_predicho == digito_esperado
#     if es_correcto:
#         aciertos_sin_ruido += 1
    
#     print(f"Dígito {digito_esperado}: Predicción={digito_predicho}, "
#           f"Salidas={[f'{p:.3f}' for p in prediccion]}, "
#           f"Correcto: {'✓' if es_correcto else '✗'}")

# precisión_sin_ruido = (aciertos_sin_ruido / len(x_train)) * 100
# print(f"\nPrecisión sin ruido: {aciertos_sin_ruido}/{len(x_train)} ({precisión_sin_ruido:.1f}%)\n")


# # PASO 6: Evaluar con ruido
# print("=" * 70)
# print("PREDICCIONES CON RUIDO (PROBABILIDAD 0.02)")
# print("=" * 70)

# probabilidad_ruido = 0.02
# num_pruebas_ruido = 10
# resultados_con_ruido = {}

# for digito_idx in range(len(x_train)):
#     aciertos_para_digito = 0
#     print(f"\nDígito {digito_idx}:")
#     print("-" * 50)
    
#     for intento in range(num_pruebas_ruido):
#         imagen_con_ruido = agregar_ruido(x_train[digito_idx], probabilidad_ruido)
#         prediccion = predecir(imagen_con_ruido, w[0], w[1], activacion_tangente, activacion_logistica)
#         digito_predicho = np.argmax(prediccion)
        
#         es_correcto = digito_predicho == digito_idx
#         if es_correcto:
#             aciertos_para_digito += 1
        
#         if intento < 3 or intento >= num_pruebas_ruido - 1:
#             print(f"  Intento {intento+1}: Predicción={digito_predicho}, Correcto: {'✓' if es_correcto else '✗'}")
#         elif intento == 3:
#             print(f"  ...")
    
#     precisión_digito = (aciertos_para_digito / num_pruebas_ruido) * 100
#     resultados_con_ruido[digito_idx] = precisión_digito
#     print(f"  Aciertos: {aciertos_para_digito}/{num_pruebas_ruido} ({precisión_digito:.1f}%)")


# # PASO 7: Resumen de resultados
# print("\n" + "=" * 70)
# print("RESUMEN DE RESULTADOS CON RUIDO (p=0.02)")
# print("=" * 70)

# for digito_idx, precisión in resultados_con_ruido.items():
#     print(f"Dígito {digito_idx}: {precisión:.1f}%")

# precisión_promedio_ruido = np.mean(list(resultados_con_ruido.values()))
# print(f"\nPrecisión promedio con ruido: {precisión_promedio_ruido:.1f}%")
# print(f"Diferencia (sin ruido - con ruido): {precisión_sin_ruido - precisión_promedio_ruido:.1f}%")

# print("\n" + "=" * 70)
# print("ANÁLISIS Y CONCLUSIONES:")
# print("=" * 70)
# print(f"""
# La red fue entrenada con:
# - 10 salidas (una para cada dígito 0-9, codificación one-hot)
# - {num_oculta} neuronas en la capa oculta
# - Función de activación tangente hiperbólica en capa oculta
# - Función de activación logística en capa de salida

# Sin ruido, la red logra una precisión del {precisión_sin_ruido:.1f}%.

# Con ruido (2% de probabilidad de invertir bits), la precisión promedio es {precisión_promedio_ruido:.1f}%.

# Esto indica que la capacidad de la red para generalizar a patrones ruidosos
# depende de cómo fue entrenada. Entrenar también con datos ruidosos mejoraría
# la robustez ante ruido en producción.
# """)


# if __name__ == "__main__":

#     w_salida = np.array([1,2,3,4,5])
#     print(w_salida[0])














