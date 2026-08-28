
#####################################################################################################
#
# Arquivo geral de simulações do Van de Vusse
#
#####################################################################################################
# %%
from dataclasses import dataclass, replace
from typing import Callable, Tuple

from pathlib import Path
from matplotlib.patches import Shadow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from numpy.typing import NDArray

directory = Path.cwd()
Array = NDArray[np.float64]
# %% 
@dataclass
class VdVParams:
    CA0: float
    CB0: float
    T0: float
    Tk0: float
    k10: float
    k20: float
    k30: float
    E1: float
    E2: float
    E3: float
    DeltaH1: float
    DeltaH2: float
    DeltaH3: float
    rho: float
    Cp: float
    K: float
    A: float
    V: float
    m: float
    Cpk: float
    CA_in: float
    
class VdV:
    
    """
    Van de Vusse CSTR (Engel & Klatt, 1993)

    A -> B -> C
    2A -> D

    Modelo não linear

    dCA_dt = F/V (CA_in - CA) - k1(T)CA - k3(T)CA^2
    dCB_dt = -F/V CB + k1(T)CA - k2(T)CB
    dT_dt = F/V (T_in - T) + Kw AR / (ρ Cp V) (Tk - T) + 1 /(ρ Cp) (k1(T) CA (-ΔH1) + k2(T)CB (-ΔH2) + k3(T) CA^2 (-ΔH3))
    dTk_dt = Q / (m Cpk) + Kw AR / (m Cpk) (T - Tk)

    sistema 2x2
    
    Manipuladas
    u = [u1, u1] = F/V, Q/KwAR
    
    Controladas
    y = [y1, y2] = Cb, T

    Disturbio
    T_in
    """
    
    def __init__(self, 
                 par: VdVParams
                 ):
        
        self.par = par
    
    def model(self, 
              y: Array, 
              u: Array,
              ) -> Array:       
        
        CA, CB, T, Tk = y
        F_V, Q_KwAR, T_in = u
        
        @staticmethod
        def arrhenius(ki, Ei, T):
            return ki * np.exp(-Ei / (T + 273.15))
        
        k1 = arrhenius(self.par.k10, self.par.E1, T) 
        k2 = arrhenius(self.par.k20, self.par.E2, T)
        k3 = arrhenius(self.par.k30, self.par.E3, T)
        
        dCAdt = F_V * (self.par.CA_in - CA) - k1 * CA - k3 * CA ** 2
        dCBdt = -F_V * CB + k1 * CA - k2 * CB
        dTdt = (
            F_V * (T_in - T)
            + self.par.K * self.par.A / (self.par.rho * self.par.Cp * self.par.V) * (Tk - T)
            + 1 / (self.par.rho * self.par.Cp)
            * (
                k1 * CA * (-self.par.DeltaH1)
                + k2 * CB * (-self.par.DeltaH2)
                + k3 * CA**2 * (-self.par.DeltaH3)
            )
        )
        dTkdt = (
            Q_KwAR * self.par.K * self.par.A / (self.par.m * self.par.Cpk)
            + self.par.K * self.par.A / (self.par.m * self.par.Cpk) * (T - Tk)
        )
        
        return np.array([dCAdt, dCBdt, dTdt, dTkdt])
    
    def jacobian(self,
            x0: Array,
            u0: Array,
            h: float = 1e-3,
            ) -> tuple[Array, Array, Array]:
        
        """Numerical evaluation of the Jacobian matrix"""
        
        n = len(x0)
        m = len(u0)
        
        J = np.zeros((n, n))
        B = np.zeros((n, m))
        
        eps = h * np.eye(n)
        epsB = h * np.eye(m)
        
        for k in range(n):
            J[:, k] = (self.model(x0+eps[k], u0) - self.model(x0-eps[k], u0)) / (2*h)
        for k in range(m):
            B[:, k] = (self.model(x0, u0+epsB[k]) - self.model(x0, u0-epsB[k])) / (2*h)    

        return J, B[:,0:m-1], B[:,m-1:m]
    
    def steady_state(self, 
              x0: Array,
              u: Array,
              ) -> Array:

        """Find steady-state response, model(x,u) = 0"""
        
        sol, _, ier, msg = fsolve(self.model, x0, args=(u,), full_output=True)
        
        if ier != 1:
            print("fsolve failed:")
            print(msg)
            print("\ntry another initial guess (x0).")
    
        return sol

    def dyn_model(self,
                  t: Array,
                  y: Array,
                  u: Array | Callable,
                  ) -> Array:
        
        U = u(t) if callable(u) else u
        
        return self.model(y, U)

    def integrate(self, 
                  t_span: Array | None,
                  y0: Array,
                  u: Array | Callable[[float], Array],
                  t_eval: Array | None = None,
                  max_step: float | None = None,
                  method: str = 'RK45',
                  rtol: float | None = 1e-9,
                  atol: float | None = 1e-9,
                  ) -> tuple[Array, Array, Array]:
        
        sol = solve_ivp(
                fun=lambda t, y: self.dyn_model(t, y, u),
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                max_step=np.inf if max_step is None else max_step,
                method=method,
                rtol=rtol,
                atol=atol
                )
        
        t_out = sol.t

        if callable(u):
            u_out = np.array([u(t_k) for t_k in t_out])
        else:
            u_out = np.tile(np.array(u), (len(t_out), 1))

        return sol.t, sol.y.T, u_out

    def step(self,
             t_span: Array,
             y0: Array,
             u0: Array,
             du: Array,
             t_eval: Array | None,
             max_step: float | None = None
             ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        
        """Plant step-test function"""
        
        u0 = np.array(u0)
        du = np.array(du)
        
        def u_step(t):
            return u0 + du if t > 0 else u0
        
        t, y, u, = self.integrate(
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            max_step=max_step,
            u=u_step
            )
        
        # return t, y, u
        return t[1:], y[1:], u[1:]

base_par = {
    "CA0": 5.1,
    "CB0": 0,
    "T0": 130,
    "Tk0": 130,
    "k10": 1.287e12,
    "k20": 1.287e12,
    "k30": 9.043e9,
    "E1": 9758.3,
    "E2": 9758.3,
    "E3": 8560,
    "DeltaH1": 4.2,
    "DeltaH2": -11.0,
    "DeltaH3": -41.85,
    "rho": 0.9342,
    "Cp": 3.01,
    "K": 4032,
    "A": 0.215,
    "V": 10.0,
    "m": 5.0,
    "Cpk": 2.0,
    "CA_in": 5.1,
}

if __name__ == '__main__':
    #%%
    def build_fir_regression_mimo(u, y, N):

        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)

        nsamples, nu = u.shape
        _, ny = y.shape

        M = nsamples - N

        # matriz de regressão
        Phi = np.zeros((M, nu * N))

        for k in range(M):

            t = k + N

            row = []

            for j in range(nu):

                row.extend(
                    u[t - N:t, j][::-1]
                )

            Phi[k] = row

        # saídas alinhadas
        Y = y[N:]

        return Phi, Y