
#####################################################################################################
#
# Arquivo geral de simulações do Van de Vusse
#
#####################################################################################################
# %%
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from numpy.typing import ArrayLike, NDArray

# %% 
@dataclass
class VdVParams:
    """
    Parameters of the van de Vusse model
    """
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
    cA_in: float
    
class VdV:
    
    """
    van de Vusse CSTR (Engel & Klatt, 1993)

    A -> B -> C
    2A -> D

    Nonlinear Model

    dcA_dt = F/V (cA_in - cA) - k1(T)cA - k3(T)cA^2
    dcB_dt = -F/V cB + k1(T)cA - k2(T)cB
    dT_dt  = F/V (T_in - T) + Kw AR / (ρ Cp V) (Tk - T) + 1 /(ρ Cp) (k1(T) cA (-ΔH1) + k2(T)cB (-ΔH2) + k3(T) cA^2 (-ΔH3))
    dTk_dt = Q / (m Cpk) + Kw AR / (m Cpk) (T - Tk)
    
    STATE VARIABLES

    x  = [cA, cB, T, Tk]

    OUTPUT VARIABLES

    y  = [cB, T]

    INPUT VARIABLES

    u  = [F/V, Q/KwAR]

    MEASURED DISTURBANCE
    
    d  = T_in
    """
    
    def __init__(self, par: VdVParams):
        self.par = par
    
    def model(self, x: np.ndarray, u: np.ndarray,) -> np.ndarray:       
        
        cA, cB, T, Tk = x
        F_V, Q_KwAR, T_in = u

        # Arrhenius equation
        @staticmethod
        def arrhenius(ki, Ei, T):
            return ki * np.exp(-Ei / (T + 273.15))
        
        k1 = arrhenius(self.par.k10, self.par.E1, T) 
        k2 = arrhenius(self.par.k20, self.par.E2, T)
        k3 = arrhenius(self.par.k30, self.par.E3, T)
        
        dcAdt = F_V * (self.par.cA_in - cA) - k1 * cA - k3 * cA ** 2
        dcBdt = -F_V * cB + k1 * cA - k2 * cB
        dTdt = (
            F_V * (T_in - T)
            + self.par.K * self.par.A / (self.par.rho * self.par.Cp * self.par.V) * (Tk - T)
            + 1 / (self.par.rho * self.par.Cp)
            * (
                k1 * cA * (-self.par.DeltaH1)
                + k2 * cB * (-self.par.DeltaH2)
                + k3 * cA**2 * (-self.par.DeltaH3)
            )
        )
        dTkdt = (
            Q_KwAR * self.par.K * self.par.A / (self.par.m * self.par.Cpk)
            + self.par.K * self.par.A / (self.par.m * self.par.Cpk) * (T - Tk)
        )

        rhs = np.array([dcAdt, dcBdt, dTdt, dTkdt])

        return rhs
    
    def jacobian(self,
            x0: np.ndarray,
            u0: np.ndarray,
            h: float = 1e-3,
            ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Numerical evaluation of the Jacobian matrix
        """
        n = len(x0)
        m = len(u0)
        
        J = np.zeros((n, n))
        B = np.zeros((n, m))
        
        eps = h * np.eye(n)
        epsB = h * np.eye(m)
        
        for k in range(n):
            J[:, k] = (self.model(x0+eps[k], u0) - self.model(x0-eps[k], u0)) / (2 * h)
        for k in range(m):
            B[:, k] = (self.model(x0, u0+epsB[k]) - self.model(x0, u0-epsB[k])) / (2 * h)    

        return J, B[:,0:m-1], B[:,m-1:m]
    
    def steady_state(self, x0: np.ndarray, u: np.ndarray,) -> NDArray[np.float64]:
        """
        Find steady-state response, model(x, u, d) = 0
        """
        
        sol, _, ier, msg = fsolve(self.model, x0, args=(u,), full_output=True)
        
        if ier != 1:
            print("fsolve failed:")
            print(msg)
            print("\ntry another initial guess (x0).")
    
        return sol

    def dyn_model(
            self,
            t: np.ndarray,
            y: np.ndarray,
            u: np.ndarray | Callable,
    ) -> NDArray[np.float64]:
        
        U = u(t) if callable(u) else u
        
        return self.model(y, U)

    def integrate(
            self, 
            t_span: ArrayLike | None,
            x0: ArrayLike,
            u: np.ndarray | Callable[[float], np.ndarray],
            t_eval: np.ndarray | None = None,
            max_step: float | None = None,
            method: str = 'RK45',
            rtol: float | None = 1e-9,
            atol: float | None = 1e-9,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Numerical integration of the model using scipy `solve_ivp`.
        """
        
        sol = solve_ivp(
                fun=lambda t, y: self.dyn_model(t, y, u),
                t_span=t_span,
                y0=x0,
                t_eval=t_eval,
                max_step=np.inf if max_step is None else max_step,
                method=method,
                rtol=rtol,
                atol=atol
                )

        if callable(u):
            u_out = np.array([u(t_k) for t_k in sol.t])
        else:
            u_out = np.tile(np.array(u), (len(sol.t), 1))

        return sol.t, sol.y.T, u_out

    def step(
            self,
            x0: ArrayLike,
            u0: ArrayLike,
            dt: float | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        
        """
        One-step integration.
        """

        if dt is None:
            dt = 1.0

        x0 = np.asarray(x0, dtype=np.float64)
        u0 = np.asarray(u0, dtype=np.float64)
        
        t, y, u, = self.integrate(
            t_span=[0, dt],
            x0=x0,
            u=u0,
            max_step=dt,
            )

        return t[-1], y[-1], u[-1]

base_par = {
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
    "cA_in": 5.1,
}