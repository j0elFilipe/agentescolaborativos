import numpy as op
import random

#FUNÇÃO PARA CRIAR O AMBIENTE
def criar_ambiente():
    matriz = op.random.choice(['L', 'B', 'T'], size = (10, 10), p = [0.5, 0.3, 0.2])
    matriz[random.randint(0,9)][random.randint(0, 9)] = 'F'

    return matriz