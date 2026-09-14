#%%
from typing import List, Deque
from collections import deque

import numpy as np
from numpy.typing import NDArray

from .models import FeedbackState, StepResponseModel
from .optimize import CostBuilder, ConstraintsBuilder, QPSolve
from .utils import Logger

class QDMC:
    """
    Quadratic Dynamic Matrix Control (QDMC).

    This class builds the QDMC controller based on the traditional step response formulation:

        Y = f + G ΔU

    where:
        Y   :   model prediciton
        f   :   system free response
        G   :   dynamic matrix
        ΔU  :   future control increments

    The optimization problem follows the quadratic programming:

        min     1/2 x^T H x + c^T x
        s.t.    A x <= b

    where:
        H   :   Hessian matrix
        c   :   linear coefficients
        A   :   inequality constraints matrix
        b   :   inequality constraints vector
        x   :   decision variables (future control increments)

    The supported constraints are:

        control increments constraints (Δu)
        input  constraints (u)
        output constraints (y)

    Parameters
    ----------
    model   :   model
                Linear model of the process
    
    Q       :   NDArray[np.float64] | list[float]
                Weights matrix of output variables

    R       :   NDArray[np.float64] | list[float]
                Weights matrix of input variables

    logger  :   Logger
                Standar logging for controller execution

    scheduling  :   dict[str, NDArray], optional
                    Scheduler for multiple models usage

    Attributes
    ----------
    ny      :   int
                Number of output (controlled) variables

    nu      :   int
                Number of input (manipulated) variables

    Np      :   int
                Size of the prediction horizon

    Nc      :   int
                Size of the prediction horizon

    qp_cost :   CostBuilder
                Quadratic cost function of the optimization problem.

    qp_constraints  :   ConstraintsBuilder
                        Constraints of the optimization problem.

    Returns
    -------
    move    :   NDArray[np.float64]
                Current control increment
                Shape(nu,)

    Notes
    -----
    The model used in the constructor already contains the information of prediction horizon Np, control horizon (Nc), number of inputs (nu) and number of outputs (ny).

    The quadratic cost function and constraints of the optimization problem are builded in the constructor. While in the execution, the cost is update with the available feedback information.

    If sheduling is used, the controller assumes a Linear Parameter-Varying (LPV) model by using linear interpolation on the multiple models informed in the scheduling dictionary. Also, the free response depends on the scheduling variable, and the cost function (Hessian matrix and linear coefficient) will be updated in every sampling.

    Examples
    --------
    >>> from Utils import Logger
    >>> from controllers import QDMC
    >>> controller = QDMC(model=my_model, Q=np.diag([100, 1]), R=np.diag([1,1]), logger=Logger())
    """
    def __init__(
            self,
            model: StepResponseModel,
            Q: NDArray[np.float64] | list[float],
            R: NDArray[np.float64] | list[float],
            logger: Logger,
            ):
        
        self.model = model
        
        self.Q = Q
        self.R = R        

        self._Np = self.model.Np
        self._Nc = self.model.Nc
        
        self._ny = self.model.ny
        self._nu = self.model.nu
        
        self._nvar = self._nu * self._Nc

        self.qp_cost = CostBuilder(self.model, self.Q, self.R)
        self.qp_constraints = ConstraintsBuilder(self.model)

        self.qp_solver = QPSolve(self._ny, self._Nc)

        self.logger = logger

        self.count = 0

        self.init_move_history()
    
    def init_move_history(self) -> None:
        self.move_history: Deque[NDArray[np.float64]] = deque([np.zeros(self._nu) for _ in range(self._Np)], maxlen=self._Np)
        self.true_move_history: Deque[NDArray[np.float64]] = deque([np.zeros(self._nu) for _ in range(self._Np)], maxlen=self._Np)

    def update_move_history(self, du_past: NDArray[np.float64], true_du_past: NDArray[np.float64]) -> None:
        self.move_history.appendleft(du_past)
        self.true_move_history.appendleft(true_du_past)
        
    def reset(self):
        self.move_history: Deque[NDArray[np.float64]] = deque([np.zeros(self._nu) for _ in range(self._Np)], maxlen=self._Np)
        self.true_move_history: Deque[NDArray[np.float64]] = deque([np.zeros(self._nu) for _ in range(self._Np)], maxlen=self._Np)
        self.count = 0 
        self.logger._reset()

    def __call__(
            self,
            y_sp: NDArray[np.float64],
            y_meas: NDArray[np.float64],
            u_meas: NDArray[np.float64],
            d_meas: NDArray[np.float64] | None = None,
            mode: List[str] | None = None,
            lbdu: NDArray[np.float64] | None = None,
            ubdu: NDArray[np.float64] | None = None,
            lbu: NDArray[np.float64] | None = None,
            ubu: NDArray[np.float64] | None = None,
            lby: NDArray[np.float64] | None = None,
            uby: NDArray[np.float64] | None = None,
            solver: str = 'quadprog'
            ):

        feedback_state = FeedbackState(y_meas=y_meas, du_past=self.move_history)
        F = self.model.free_response(feedback_state)
        self.eu = F

        c = self.qp_cost.update(
                        y_sp=y_sp,
                        F=F,
                        mode=mode,
                        lby=lby,
                        uby=uby
                        )
        
        b = self.qp_constraints.update(
                        F=F,
                        uprev=u_meas,
                        lbu=lbu,
                        ubu=ubu,
                        lby=lby,
                        uby=uby,
                        )
        
        sol, cost = self.qp_solver(
                        P=self.qp_cost.H,
                        q=c,
                        G=self.qp_constraints.A,
                        h=b,
                        lb=lbdu,
                        ub=ubdu,
                        solver=solver
                        )
        
        du_full = sol[:self._nvar]
        du = du_full[::self._Nc]
        # self.du_full = du_full
        s = sol[self._nvar:]

        if self.count > 0:
            true_du = u_meas - self.u_meas_prev
        else:
            self.u_meas_prev = u_meas
            true_du = np.zeros(self._nu, dtype=np.float64)

        self.u_meas_prev = u_meas.copy()
        
        self.update_move_history(du, true_du)

        self.count += 1

        if self.logger is not None:

            self.logger.write(
                y_sp=y_sp,
                y_meas=y_meas,
                y_pred=F.T[0],
                u_meas=u_meas,
                move=du,
                cost=cost,
                lbu=lbu,
                ubu=ubu,
                lby=lby,
                uby=uby,
            )

        return du, s