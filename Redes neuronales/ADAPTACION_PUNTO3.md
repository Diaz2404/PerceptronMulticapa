# Adaptación del Perceptrón Multicapa a 10 Salidas

## ¿Qué cambió?

Tu código del perceptrón multicapa **ya estaba preparado para múltiples salidas**. No necesitaba cambios en las matemáticas (delta, retropropagación, actualización de pesos). Solo necesitábamos cambiar **cómo preparamos los datos**.

## Cambios Realizados

### 1. **One-Hot Encoding** (La mayor diferencia)

**Antes (punto2: 1 salida):**
```python
y_train = [1, -1, 1, -1, 1, -1, 1, -1]  # Forma: (8,)
# Una única salida: 1 si par, -1 si impar
```

**Ahora (punto3: 10 salidas):**
```python
salidas_one_hot = np.eye(10)  # Matriz identidad 10×10
# Para cada dígito, una salida diferente:
# Dígito 0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Dígito 1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# Dígito 2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# ... y así sucesivamente
```

### 2. **Parámetro de la Red**
```python
num_salida = 10  # CAMBIO: era num_salida=1 en punto2
```

### 3. **Función de Activación de Salida**
Cambié a **logística** (devuelve valores entre 0 y 1) porque one-hot encodes requiere eso:
```python
funcion_activacion_salida=activacion_logistica  # Antes: tanh
```

### 4. **Función para Agregar Ruido**
Se agregó una función que invierte bits con probabilidad 0.02:
```python
def agregar_ruido(imagen, probabilidad_ruido=0.02):
    imagen_con_ruido = imagen.copy()
    for i in range(len(imagen_con_ruido)):
        if np.random.rand() < probabilidad_ruido:
            imagen_con_ruido[i] = 1 - imagen_con_ruido[i]  # Invertir bit
    return imagen_con_ruido
```

## ¿Por Qué el Código YA Funcionaba sin Cambios?

Las funciones matemáticas ya trabajaban con **vectores**, no con escalares.

### Ejemplo: `calcular_delta_salida()`

```python
def calcular_delta_salida(y_pred, y_true, h_salida, derivada_activacion):
    delta_salida = (y_true - y_pred) * derivada_activacion(h_salida)
    return delta_salida
```

- Con 1 salida: `y_true` es un número (ej: 1), `y_pred` es un número
- Con 10 salidas: `y_true` es un array de 10 (ej: [0, 0, 1, 0, ...]), `y_pred` es un array de 10
- **La fórmula funciona exactamente igual** porque NumPy hace la multiplicación elemento a elemento

### Ejemplo: `actualizar_pesos_salida()`

```python
def actualizar_pesos_salida(w, delta, o_anterior, n):
    o_anterior_con_bias = np.append(o_anterior, 1)
    return w + n * delta * o_anterior_con_bias
```

- Con 1 salida: `w` es un vector, `delta` es un número
- Con 10 salidas: `w` es una matriz 10×33, `delta` es un vector de 10
- **NumPy expande automáticamente** la multiplicación: cada fila de w se multiplica por el delta correspondiente

## Fórmulas (Sin Cambios Matemáticos)

### Delta de la capa de salida
$$\delta_i = (y_i - \hat{y}_i) \cdot \sigma'(h_i)$$

Donde $i$ va de 0 a 9 (en lugar de solo 0 a 1).

### Delta de la capa oculta (idéntica)
$$\delta_j = \sigma'(h_j) \sum_{i=0}^{9} w_{ij} \delta_i$$

Simplemente sumamos sobre 10 salidas en lugar de 1.

### Actualización de pesos (idéntica)
$$w_{nuevo} = w_{viejo} + n \cdot \delta \cdot entrada$$

Funciona para matrices de cualquier tamaño.

## Cómo Usar el Punto 3

### Ejecutar el entrenamiento y prueba:
```bash
conda activate perceptron-multicapa
python src/punto3.py
```

### Qué pasa:
1. Entrena la red con los 10 dígitos (one-hot)
2. Prueba sin ruido: muestra precisión (idealmente 100%)
3. Prueba con ruido (p=0.02): muestra precisión con ruido para cada dígito
4. Compara: "¿Cuánto empeora la red con ruido?"

### Resultados esperados:
- **Sin ruido**: ~100% de precisión (entrenó con esos datos)
- **Con ruido**: Varía, pero típicamente 50-90% según el dígito
- **Diferencia**: Muestra cuán robusta es la red ante ruido

## Modificaciones Fáciles

Si querés ajustar el comportamiento:

```python
# Cambiar tasa de aprendizaje
n = 0.05  # Más lento pero más estable

# Cambiar número de neuronas ocultas
num_oculta = 64  # Más neuronas = más capacidad (pero más lento)

# Cambiar probabilidad de ruido
probabilidad_ruido = 0.05  # Más ruido = más difícil

# Cambiar número de iteraciones de entrenamiento
cota = 10000  # Más iteraciones = mejor convergencia (pero más tiempo)
```

## Conclusión

**La mágica del perceptrón multicapa es que el algoritmo fundamentalmente NO DEPENDE del número de salidas.** Solo necesitamos:
1. Preparar los datos correctamente (one-hot encoding)
2. Ajustar la dimensión de la matriz de pesos inicial
3. Elegir funciones de activación adecuadas

Todo lo demás (delta, retropropagación, actualización) funciona igual para 1 salida, 10 salidas o 1000 salidas.
