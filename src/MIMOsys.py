from typing import Callable, overload, Union
from collections import deque

import numpy as np
from numpy.typing import NDArray, ArrayLike

class DiscreteFOPDT:
    def __init__(self, K: float, tau: float, theta: float, DT: float):

        self.a = np.exp(-DT / tau)
        self.b = K * (1 - self.a)
        
        self.d = int(round(theta / DT))
        
        self.u_buffer = deque([0.0] * (self.d + 1), maxlen=self.d + 1)
        
        self.x = 0.0

    def step(self, u_k: float) -> float:
        """Propaga o estado em 1 passo de tempo"""
        self.u_buffer.append(u_k)
        
        u_delayed = self.u_buffer[0]
        
        self.x = self.a * self.x + self.b * u_delayed
        
        return self.x

class DiscreteMIMO:
    def __init__(self, G_discrete):
        self.G = np.asarray(G_discrete, dtype=object)
        self.ny, self.nu = self.G.shape
        self.y = np.zeros(self.ny)

    def step(self, u_k: list) -> np.ndarray:
        """Propaga todos os blocos e soma as saídas"""
        self.y = np.zeros(self.ny)
        
        for i in range(self.ny):
            for j in range(self.nu):
                self.y[i] += self.G[i, j].step(u_k[j])
                
        return self.y

def Wood_Berry(DT=1.0):

    g11 = DiscreteFOPDT(K=12.8, tau=16.7, theta=1, DT=DT)
    g12 = DiscreteFOPDT(K=-18.9, tau=21.0, theta=3, DT=DT)
    g21 = DiscreteFOPDT(K=6.6, tau=10.9, theta=7, DT=DT)
    g22 = DiscreteFOPDT(K=-19.4, tau=14.4, theta=3, DT=DT)

    G = [
    [g11, g12],
    [g21, g22]
    ]

    mimo = DiscreteMIMO(G)

    return mimo