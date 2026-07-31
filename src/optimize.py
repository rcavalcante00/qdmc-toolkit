# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:13:26 2026

@author: rafae
"""
from typing import Literal, List, Tuple, Deque

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.linalg import block_diag
from qpsolvers import solve_qp, solve_problem
from .models import ProcessModel

class CostBuilder:
            
    """
    Cost build for model predictive controllers in MIMO systems.

    This class builds the Hessian matrix and the linear coefficients
    of the quadratic optimization problem for model predictive control
    based on linear models.

    The quadratic problem is set as follows:

    min     1/2 x^T H x + c^T x
    s.t.    A x <= b

    where:
        H: Hessian matrix
        c: linear coefficients
        A: inequality constraints matrix
        b: inequality constraints vector
        x: decision variables (future control increments)
    
    Parameters
    ----------
    model : model
            Linear model of the process.
    
    Q     : NDArray[np.float64] | List[float]
            Weights matrix for controlled variables.

    R     : NDArray[np.float64] | List[float]
            Suppression factor for manipulated variables.

    slack_weights  : NDArray[np.float64], Literal["auto"]
                     Weight of slack variables for soft constraints

    Attributes
    ----------
    H      : NDArray[np.float64]
             Hessian matrix
             Shape: (nu*Nc + 2*ny, nu*Nc + 2*ny)
    
    c_init : NDArray[np.float64]
             Precomputed vector of linear coefficients
             Shape: (nu*Nc, ny*Np)

    Notes
    -----
    The builder standard assumes a controlled system
    with bounds for output variables, for which the
    Hessian matrix is augmented considering slack variables
    for soft constraints.

    The augmented Hessian is structured as:

    [H    0]
    [0    H_slack]

    Some properties such as prediction and control horizon
    are inherited from the model used in the constructor.

    Examples
    --------
    >>> qp_cost = CostBuilder(model=my_model, Q=np.eye(2), R=np.eye(2))
    >>> H = qp._H
    >>> print(H.shape)

    """
    
    def __init__(
            self,
            model: ProcessModel,
            Q: NDArray[np.float64] | List[float],
            R: NDArray[np.float64] | List[float],
            slack_weights: NDArray[np.float64] | Literal["auto"] = "auto"
            ):
        
        self.model = model
        self.ny = self.model.ny
        self.Np = self.model.Np
        self.Nc = self.model.Nc

        self.Q = np.diag(Q) if np.ndim(Q) == 1 else np.array(Q, dtype=np.float64)
        self.R = np.diag(R) if np.ndim(R) == 1 else np.array(R, dtype=np.float64)
        
        self._slack_weights = (
            slack_weights if slack_weights != "auto"
            else 1e5 * np.ones(self.ny) * np.mean(np.diag(self.Q))
            )
        
        self.H = self.build_hessian(self.model.G)
        self.c_init = self.precompute_c(self.model.G)
    
    def precompute_c(
            self,
            G: NDArray[np.float64]
            ) -> NDArray[np.float64]:
        
        return G.T @ np.kron(self.Q, np.eye(self.Np))
    
    def build_hessian(
            self,
            G: NDArray[np.float64],
            ) -> NDArray[np.float64]:
        
        G = G
        Q = self.Q
        R = self.R
        
        Np = self.Np
        Nc = self.Nc
        
        Q_matrix = np.kron(Q, np.eye(Np))
        R_matrix = np.kron(R, np.eye(Nc))
        
        slack_weights = self._slack_weights
        
        H = 2 * (G.T @ Q_matrix @ G + R_matrix)        
        H = 0.5 * (H + H.T)
        
        # --------------------------------------------
        # Augmentation of Hessian with slack variables
        # --------------------------------------------
                
        if self._slack_weights is not None:

            H_aug = block_diag(
                        H, np.diag(
                            np.repeat(
                                slack_weights, 2)
                            )
                        )
            
            return H_aug
        
        return H
    
    def update(
            self,
            y_sp: NDArray[np.float64],
            F: NDArray[np.float64],
            mode: List[str] | None = None,
            lby: NDArray[np.float64] | None = None,
            uby: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        
        """
        Update the Quadratic Programming problem by recalculating c vector.
        
        Parameters
        ----------
        y_sp : NDArray[np.float64]
               Desired output setpoints.
               Shape: (ny,)

        F    : NDArray[np.float64]
               Free response vector over the prediction horizon.
               Shape: (ny, Np)
            
        mode : List[str], optional
               Control mode assigned to each controlled variable.
               For zone control, the target trajectory is computed by clipping
               the free response within the admissible output bounds.
            
               mode == 'setpoint' corresponds to standard tracking control.
               mode == 'zone' corresponds to zone control.
            
        lby  : NDArray[np.float64], optional
               lower bounds of the outputs.
               Shape: (ny,)
            
        uby  : NDArray[np.float64], optional
               upper bounds of the outputs.
               Shape: (ny,)

        Returns
        -------
        c : NDArray[np.float64]
            Updated c vector.
            Shape: (nu*Nc, ny*Np)
        
        Notes
        -----
        The free response (F) carries the information
        of output measurements and past move increments necessary
        for updating the optimization problem.

        """
        
        ny = self.ny
        Np = self.Np
        
        y_sp = np.asarray(y_sp, dtype=np.float64)
        
        if mode is None:
            mode = ['setpoint'] * ny
        else:
            if len(mode) != ny:
                raise ValueError(
                    f"mode must contain exactly {ny} entries "
                    f"(received {len(mode)})."
                )
        
        y_target = np.repeat(y_sp[:, None], Np, axis=1)
        
        for i in range(ny):            
            if mode is not None:
                if mode[i] == 'setpoint':                
                    pass
                    
                elif mode[i] == 'zone':                
                    y_free_i = F[i]
                    
                    y_low = -np.inf if lby is None else lby[i]
                    y_high = np.inf if uby is None else uby[i]
                    
                    if y_low > y_high:
                        raise ValueError(
                            f"lby[{i}] must be <= uby[{i}]"
                            )
                        
                    clipped = np.clip(y_free_i, y_low, y_high)

                    tol = 1e-3 * (1 + abs(y_low) + abs(y_high))
                    inside = (y_free_i >= y_low - tol) & (y_free_i <= y_high + tol)

                    y_target[i] = np.where(inside, y_free_i, clipped)
                
                else:                    
                    raise ValueError(
                f"Unknown mode: {mode[i]}")

        e = (F - y_target).ravel()
        
        c = 2 * (self.c_init @ e)
        
        # --------------------------------------------------------
        # Augmentation of coefficients vector with slack variables
        # --------------------------------------------------------

        c_aug = np.concatenate([c, np.zeros(2 * ny)])

        self.c = c_aug
            
        return c_aug
#%%
class ConstraintsBuilder:
    """
    Build linear inequality constraints for MPC optimization problem.

    The constraints follow the form:

        A x <= b

    where:

        A : inequality matrix
        b : inequality vector
        x : decision variables

    the decision vector is:

        x = [ΔU, s]^T

    where:

        ΔU : stacked control increments over the control horizon
        s  : slack variables for soft output constraints

    the slacks are defined for lower and upper bounds:
    
        s = [s_lower, s_upper]

    and positives:

         s_upper  >= 0
        -s_lower  <= 0

    The class generates constraints associated with:

    - control increments (hard)
    - input variables (hard)
    - output variables (soft)

    Parameters
    ----------
    model: model
           Linear model of the process.

    Attributes
    ----------
    nvar : int
        Number of control increment decision variables.

    nslack : int
        Number of slack variables.

    A : NDArray[np.float64]
        Global inequality constraints matrix.

    Notes
    -----
    The optimization vector is defined as:

    ΔU = [
        Δu₁(0), Δu₁(1), ..., Δu₁(Nc-1),
        Δu₂(0), Δu₂(1), ..., Δu₂(Nc-1),
        ...
        Δuₙᵤ(0), ..., Δuₙᵤ(Nc-1)
        ]^T

    where increments are stacked input-wise
    (grouped by manipulated variable).

    Predicted outputs are computed as:

        Y = F + GΔU

    and constraints are assembled in the standard form:

        A x <= b

    If a set of constraints is not defined, infinity values will be assumed, for example:

    if lby = None, the lower bounds of controlled variables will be set as -np.inf.

    """

    def __init__(
            self,
            model,
            ):
        
        self._model = model
        self._nu = self._model.nu
        self._ny = self._model.ny
        self._Np = self._model.Np
        self._Nc = self._model.Nc

        self._G = self._model.G
        
        self.nvar = self._nu * self._Nc
        self.nslack = 2 * self._ny
        self.I = np.eye(self.nvar)

        self.A = self._build_A()

    def _validate_input(self,
                        input_: NDArray[np.float64] | None,
                        expected: int,
                        name: str
                        ) -> str | None:
        
        if input_ is None:
            pass
        else:
            if len(input_) != int(expected):
                raise ValueError(f"{name} must have length {self._nu}, got {len(input_)}.")
            
    def _asarray(
            self,
            *args: ArrayLike | None,
            ) -> tuple[NDArray[np.float64] | None, ...]:
        
        return tuple(
            None if x is None else np.asarray(x, dtype=np.float64)
                for x in args)
        
    def _A_u(self) -> NDArray[np.float64]:

        """
        Build inequality (hard) constraints for input accumulation.
        
        Input constraints are formulated as:

            lbu <= u <= ubu

        and converted to inequality form.

        The general structure of A_u matrix is given by:

        u(k + Nc)   = u(k - 1) + future control increments.

        u(k)        = u(k - 1) + Δu(k)
        u(k + 1)    = u(k - 1) + Δu(k) + Δu(k+1)
        u(k + 2)    = u(k - 1) + Δu(k) + Δu(k+1) + Δu(k+2)
                    .
                    .
                    .
        u(k + Nc)   = u(k - 1) + ... + Δu(k + Nc - 1)

        Which can be written as:

        U = N ΔU + 1 u(k - 1)

        Where ΔU = [Δu(k), Δu(k + 1), ..., Δu(k + Nc - 1)] and
        N is a lower triangular matrix of the form:

                [1 0 0 ... 0 0 0]
                [1 1 0 ... 0 0 0]
        N   =   [1 1 1 ... 0 0 0]
                [1 1 1 ... 1 0 0]
                [1 1 1 ... 1 1 0]
                [1 1 1 ... 1 1 1]
        
        The final form of input inequality constraints matrix will be:

        u_min <= u <= u_max

        u_min <= N ΔU + 1 u(k - 1) <= u_max

        - N ΔU <= - u_min + 1 u(k - 1)
          N ΔU <=   u_max - 1 u(k - 1)

        A_u = [- N]
              [  N]

        The shape of A_u will be (2*nu*Nc , nu*Nc).

        Note: For MIMO systems, N matrix can be obtained by using the kronecker product.

        Considering slack variables,
    
        A_u = [- N  0]
              [  N  0]
               
        here 0 denotes a matrix of zeros with shape (2*nu*Nc, nslack),
        where nslack = 2*ny in the current implementation.

        Returns
        -------
        NDArray[np.float64]
            Input constraint matrix.
            Shape:
                (2*nu*Nc, nvar + nslack)
        """
        
        N_coeff = np.tril(np.ones((self._Nc, self._Nc)))
        N = np.kron(np.eye(self._nu), N_coeff)

        A_u = np.vstack([-N, N])
        Zeros = np.zeros((2 * self.nvar, self.nslack))

        A = np.hstack([A_u, Zeros], dtype=np.float64)

        return A
    
    def _A_y(self) -> NDArray[np.float64]:

        """
        Build soft output constraints using slack variables.

        Output constraints are formulated as:

            lby <= y <= uby

        and converted to inequality form with independent slack
        variables for lower and upper violations.

        The structure of A_y matrix is given by:

        y_min - s_lower <= G Δu + F <= y_max + s_upper
        
        - G Δu <= - y_min + s_lower + F
          G Δu <=   y_max + s_upper - F

        A_y = [ G]
              [-G]

        The shape of A_y will be (2*ny*Np, nu*Nc).

        The slack variables matrix is defined as:

        S_y = [S_lower]
              [S_upper]

        The shape of S_y will be (2*ny*Np, nslack)

        The matrix of soft constraints is:

        A_soft = [A_y  S_y]

        A is augmented with slacks contribution

        A = [A_soft ]
            [A_slack]

        Returns
        -------
        NDArray[np.float64]
            Output constraint matrix including slack variables.

        Shape:
            (2*ny*Np + nslack, nvar + nslack)
        """

        m = self._ny * self._Np
        nslack = self.nslack

        A_y = np.vstack([-self._G, self._G])
        slacks = []

        S_lower = np.zeros((m, nslack))
        S_upper = np.zeros((m, nslack))

        for i in range(self._ny):
            rows = slice(i * self._Np, (i + 1) * self._Np)
            S_lower[rows, i] = -1
            S_upper[rows, self._ny + i] = -1
            
        slacks.append(S_lower)
        slacks.append(S_upper)
        S_y = np.vstack(slacks)

        A_soft = np.hstack([A_y, S_y])
        A_slack = np.hstack([np.zeros((nslack, self.nvar)), -np.eye(nslack)])
        
        return np.vstack([A_soft, A_slack], dtype=np.float64)
    
    def _build_A(self) -> NDArray[np.float64]:
        """
        Returns
        -------
            Inequality constraints matrix.
            Shape (2*ny*Np + 2*nu*Nc + nslack, nvar + nslack)
        """
        return np.vstack([self._A_u(), self._A_y()])
    
    def _b_u(
            self,
            uprev: NDArray[np.float64], 
            lbu: NDArray[np.float64] | None = None,
            ubu: NDArray[np.float64] | None = None,
            ) -> NDArray[np.float64]:
        
        self._validate_input(lbu, self._nu, 'lbu')
        self._validate_input(ubu, self._nu, 'ubu')

        lbu, ubu = self._asarray(lbu, ubu)

        b_u = []

        u_lb = lbu if lbu is not None else np.full(self._nu, -np.inf)
        u_ub = ubu if ubu is not None else np.full(self._nu, np.inf)

        b_u.append(np.repeat(uprev, self._Nc) - np.repeat(u_lb, self._Nc))
        b_u.append(np.repeat(u_ub, self._Nc) - np.repeat(uprev, self._Nc))

        b = np.concatenate(b_u)

        return b
    
    def _b_y(
            self,
            F: NDArray[np.float64],
            lby: NDArray[np.float64] | None = None,
            uby: NDArray[np.float64] | None = None
            ) -> NDArray[np.float64]:
        
        self._validate_input(lby, self._ny, 'lby')
        self._validate_input(uby, self._ny, 'uby')

        lby, uby = self._asarray(lby, uby)
    
        b_y = []

        y_lb = lby if lby is not None else np.full(self._ny, -np.inf)
        y_ub = uby if uby is not None else np.full(self._ny, np.inf)

        b_y.append(-(y_lb[:, None] - F))
        b_y.append(y_ub[:, None] - F)

        b = np.concatenate(b_y)

        return b

    def _build_b(
            self,
            F: NDArray[np.float64],
            uprev: NDArray[np.float64],
            lbu: NDArray[np.float64] | None = None,
            ubu: NDArray[np.float64] | None = None,
            lby: NDArray[np.float64] | None = None,
            uby: NDArray[np.float64] | None = None
            ) -> NDArray[np.float64]:

        b_u = self._b_u(
                    uprev=uprev,
                    lbu=lbu,
                    ubu=ubu
                    )
        
        b_y =  self._b_y(
                    F=F,
                    lby=lby,
                    uby=uby
                    ).ravel()

        b_0 = np.zeros(self.nslack)

        b = np.concatenate([b_u, b_y, b_0])

        return b
        
    def update(
            self,
            F: NDArray[np.float64],
            uprev: NDArray[np.float64],
            lbu: NDArray[np.float64] | None = None,
            ubu: NDArray[np.float64] | None = None,
            lby: NDArray[np.float64] | None = None,
            uby: NDArray[np.float64] | None = None
            ) -> NDArray[np.float64]:
        """
        Update the inequality constraints vector (b).

        Parameters
        ----------
        F : NDArray[np.float64]
            Free response vector over the prediction horizon.
            Shape: (ny, Np)

        uprev : NDArray[np.float64]
                Previous control input.
                Shape: (nu,)
        
        Optional Parameters
        -------------------

        lbdu, ubdu : NDArray[np.float64]
                     Lower/upper bounds for control increments.
                     Shape: (nu,)

        lbu, ubu : NDArray[np.float64]
                   Lower/upper bounds for manipulated variables.
                   Shape: (nu,)

        lby, uby : NDArray[np.float64]
                   Lower/upper bounds for controlled variables.
                   Shape: (ny,)

        Returns
        -------
        NDArray[np.float64]
            Updated inequality vector.
            Shape: (nu*Nc + nslack)
        """

        if uprev is None:
            raise ValueError("feedback of input variables must be informed.")
        
        self._validate_input(uprev, self._nu, 'uprev')

        return self._build_b(
                        F=F,
                        uprev=uprev,
                        lbu=lbu,
                        ubu=ubu,
                        lby=lby,
                        uby=uby
                        )
#%%
class QPSolve:
    
    def __init__(self, ny: int, Nc: int):
        self._Nc = Nc
        self._ny = ny

    def __call__(
            self,
            P  : NDArray[np.float64],
            q  : NDArray[np.float64],
            G  : NDArray[np.float64] | None = None,
            h  : NDArray[np.float64] | None = None,
            lb : NDArray[np.float64] | None = None,
            ub : NDArray[np.float64] | None = None,
            solver : str | None = None,
            initvals : NDArray[np.float64] | None = None,
            verbose : bool = False
            ) -> Tuple[NDArray[np.float64], NDArray[np.float64]] | Tuple[NDArray[np.float64], float]:
        
        nvar = q.size
        
        if lb is not None:
            lb_du = np.repeat(lb, self._Nc)
            lb_slack = np.zeros(2 * self._ny)
            lb = np.concatenate([lb_du, lb_slack])

        if ub is not None:
            ub_du = np.repeat(ub, self._Nc)
            ub_slack = np.full(2 * self._ny, np.inf)
            ub = np.concatenate([ub_du, ub_slack])
        
        try:
            sol = solve_qp(
                    P=P,
                    q=q, 
                    G=G, 
                    h=h,
                    lb=lb,
                    ub=ub,
                    solver=solver,
                    initvals=initvals,
                    verbose=verbose
                    )
        
            if sol is None:
                print("Warning: QP solver returned no solution.")
                print("Possible causes: Infeasible constraints or numerical issues")
                print("Returning zero control increment.")
        
                return np.zeros(nvar), np.nan
            
        except Exception as e:
            print(f"QP solver error: {e}")
            print("Returning zero control action")

            return np.zeros(nvar), np.nan
        
        cost = 0.5 * sol.T @ P @ sol + q.T @ sol

        return sol, cost

        

# if __name__ == '__main__':
    
#     VdV_step = np.load(r"C:\Users\rafae\OneDrive\Tese\Controle\Studies\VdV\step20.npz")
#     Y = VdV_step['Y']
#     du = VdV_step['du']
    
#     Np = 160
#     Nc = 8
    
#     modelo = StepResponseModel(y=Y, step_amplitudes=du, Np=Np, Nc=Nc)
    
#     Q = [100, 1]
#     R = [1, 1]
    
#     qp = QPBuilder(model=modelo,
#                    Q=Q,
#                    R=R,
#                    Np=Np,
#                    Nc=Nc,
#                    )