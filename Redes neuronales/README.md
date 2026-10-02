# Perceptron Multicapa

Este proyecto está preparado para trabajar con Python y, por ahora, solo necesita la librería `numpy`.

La instalación de dependencias se maneja con el archivo `environment.yml`.

## Requisitos previos

Antes de empezar, instalá una de estas opciones:

1. Miniconda
2. Anaconda
3. Mambaforge

Si ya tenés uno de esos gestores de entornos instalado, podés pasar al siguiente paso.

## Instalación paso por paso

### 1. Abrir una terminal en la carpeta del proyecto

Ubicate en la raíz del proyecto, donde está el archivo `environment.yml`.

### 2. Crear el entorno

Ejecutá este comando:

```bash
conda env create -f environment.yml
```

Esto va a crear un entorno llamado `perceptron-multicapa` e instalar `numpy`.

### 3. Activar el entorno

Una vez creado, activalo con:

```bash
conda activate perceptron-multicapa
```

### 4. Verificar que numpy quedó instalado

Probá esto:

```bash
python -c "import numpy; print(numpy.__version__)"
```

Si imprime una versión, la instalación fue correcta.

### 5. Usar el entorno en VS Code

Si trabajás desde Visual Studio Code, elegí el intérprete del entorno creado.

En general, tenés que seleccionar el entorno `perceptron-multicapa` como intérprete de Python.

## Cómo ejecutar el proyecto

Con el entorno activado, podés correr tus scripts normalmente. Por ejemplo:

```bash
python src/punto1.py
```

o

```bash
python src/punto2.py
```

## Actualizar dependencias

Si en el futuro agregás más librerías al proyecto, editá `environment.yml` y luego ejecutá:

```bash
conda env update -f environment.yml --prune
```

## Reinstalar desde cero

Si querés borrar el entorno y crearlo otra vez:

```bash
conda env remove -n perceptron-multicapa
```

Después repetí los pasos de instalación.

## Estructura general

- `src/`: código fuente del proyecto
- `data/`: datos de entrada
- `environment.yml`: dependencias del entorno
- `.gitignore`: archivos que Git no debe seguir
