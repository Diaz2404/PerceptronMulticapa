import numpy as np

def activacion_lineal(h):
    return h

def activacion_lineal_derivada(h):
    return 1

# logistica: valores entre 0 y 1
def activacion_logistica(h, B=1):
    return 1 / (1 + np.exp(-2 * B * h))  # g(h) = 1 / (1 + e^(-2Bh))  donde B es un parametro de ajuste

def activacion_logistica_derivada(h, B=1):
    g = activacion_logistica
    return 2 * B * g(h) * (1 - g(h))  # g´(h) = 2Bg(h)(1 - g(h))  donde g(h) = 1 / (1 + e^(-2Bh))


# tangente hiperbólica: valores entre -1 y 1
def activacion_tangente(h, B=1):
    return np.tanh(B * h)  # g(h) = tanh(Bh)

def activacion_tangente_derivada(h, B=1):
    g = activacion_tangente
    return B * (1 - g(h) ** 2)  # g´(h) = B(1 - g(h)^2)  donde g(h) = tanh(Bh)


