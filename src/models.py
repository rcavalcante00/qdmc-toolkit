"""
This code define some linear models for Model Predictive Control applications. The models are used to build the Hessian matrix and the linear vector of the quadratic optimization problem.


@author: rafae
"""
import warnings
from typing import Protocol, Deque, cast, overload
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import toeplitz
from scipy.signal import cont2discrete

@dataclass
class FeedbackState:
    """
    Encapsulates the current state of the plant at time k. Used to standardize data input into prediction models.

    Attributes
    ----------
    y_meas  :   NDArray[np.float64]
                Current measurement of output variables
                Shape(ny,)

    du_past :   NDArray[np.float64] | Deque[NDArray[np.float64]] | Optional
                Full history of the control increments, used only in step response-based models.
                Shape(Np,nu)

    u_past  :   NDArray[np.float64] | Optional
                Last value of the input variables, generally the current measurement, used in state space models.   
                Shape(nu,)

    rho     :   NDArray[np.float64] | float | Optional
                Scheduling variable for Linear Parameter-Varying (LPV) models.

    Notes
    -----
    the ``du_past`` vector is ordered as:
    
            du_past[0] = Δu(k-1)
            du_past[1] = Δu(k-2)
                    .
                    .
                    .
            du_past[Np] = Δu(k-Np-1)
    """
    y_meas: NDArray[np.float64]
    du_past: NDArray[np.float64] | Deque[NDArray[np.float64]] | None = None
    u_past: NDArray[np.float64] | None = None
    rho: NDArray[np.float64] | float | None = None

class ProcessModel(Protocol):
    nu: int
    ny: int
    Np: int
    Nc: int
    G:  NDArray[np.float64]
    # S:  NDArray[np.float64]
    deltaS: NDArray[np.float64]

    # def free_response(
    #         self,
    #         feedback: FeedbackState
    #         ) -> NDArray[np.float64]:
    #     ...

    # def interpolate(
    #         self,
    #         rho: float
    #         ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    #     ...

def compute_free_response(deltaS: NDArray[np.float64],
                          feedback: FeedbackState) -> NDArray[np.float64]:
    ny = deltaS.shape[0]
    y_meas = feedback.y_meas
    du_past = np.asarray(feedback.du_past, dtype=np.float64)
    
    f = np.einsum('ijpm,mj->ip', deltaS, du_past)

    y_model = f[:,0]
    e = y_meas.reshape(ny) - y_model

    return f + e[:,None]

class DynamicMatrix:

    def __init__(
            self,
            Np: int,
            Nc: int,
            y: NDArray[np.float64],
            step_amp: NDArray[np.float64]
            ) -> None:

        self.Np = int(Np)
        self.Nc = int(Nc)

        self.y = self.fit_y_to_horizon(y, Np) if y.shape[1] != Np else np.asarray(y, dtype=np.float64) 
        self.step_amp = np.asarray(step_amp, dtype=np.float64)

        self.nu: int = self.step_amp.size
        self.ny: int = y.shape[0] // self.nu

        self.validate_inputs()

        self.G, self.S = self.build_dynamic_matrix()
        
        self.deltaS = self.build_delta_s(S=self.S)

    def validate_inputs(self):

        if self.Np <= 0 or self.Nc <= 0:
            raise ValueError("Np and Nc must be positive integers.")
    
        if self.Nc > self.Np:
            raise ValueError("Control horizon must be equal or less than prediction horizon (Nc <= Np).")
    
        if self.step_amp.ndim != 1:
            raise ValueError("step_amp must be a 1D array.")
    
        if np.any(self.step_amp == 0):
            raise ValueError("step_amp contains zero(s), cannot normalize step response.")
    
        if self.y.ndim != 2:
            raise ValueError(
                "y must be a 2D array: shape (ny*nu, Nsteps)"
            )
        
    def fit_y_to_horizon(
            self,
            y: NDArray[np.float64], 
            Np: int,
            ) -> NDArray[np.float64]:      
        y = np.asarray(y, dtype=np.float64)
        Yf = []     
        for row in y:    
            if len(row) < Np:       
                row = np.pad(
                    row,
                    (0, Np - len(row)),
                    mode='edge'
                    )           
            else:
                row = row[:Np]                
            Yf.append(row)         
        return np.asarray(Yf, dtype=np.float64)
    
    def build_dynamic_matrix(
                self,
                ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:    
            y = self.y
            step_amp = self.step_amp
            
            def build_toeplitz(s):
                col = s[:self.Np]
                row = np.zeros(self.Nc)
                row[0] = s[0]
                return toeplitz(col, row)
            
            G_blocks = []
            S = np.zeros((self.ny * self.nu, self.Np), dtype=np.float64)
            
            for i in range(self.ny):
                row_blocks = []
                for j in range(self.nu):                  
                    idx = i * self.nu + j                  
                    if step_amp[j] == 0:                  
                        raise RuntimeError(f'step_amplitudes[{j}] is zero, cannot normalize step response.')
                        
                    s = (y[idx] - y[idx][0]) / step_amp[j]                 
                    S[idx] = s
                    Gij = build_toeplitz(s)               
                    row_blocks.append(Gij)                                  
                G_blocks.append(row_blocks)
                
            G_rows = [np.hstack(row) for row in G_blocks]       
            G = np.vstack(G_rows)
            
            return G, S
    
    def build_delta_s(self, S:NDArray[np.float64]) -> NDArray[np.float64]:   
        deltaS = np.zeros(
            (self.ny, self.nu, self.Np, self.Np),
            dtype=np.float64
        ) 
        for i in range(self.ny):    
            for j in range(self.nu):    
                s_ij = S[i * self.nu + j]    
                for p in range(self.Np):    
                    for m in range(self.Np):    
                        if p + m + 1 < len(s_ij):    
                            deltaS[i, j, p, m] = (
                                s_ij[p + m + 1]
                                - s_ij[m]
                            )
                        else:
                            deltaS[i, j, p, m] = s_ij[-1] - s_ij[m]
                            
        return deltaS
    
    def save_npz(self, filepath: str):
        np.savez(filepath,
                 G=self.G,
                 S=self.S,
                 deltaS=self.deltaS,
                 Np=self.Np,
                 Nc=self.Nc,
                 ny=self.ny,
                 nu=self.nu
                 ) 

class StepResponseModel:
    """
    Step Response Model for MIMO systems (DMC/QDMC formulations).

    This class builds the LTI Step Response Model by calling the DynamicMatrix class.

    The model follows the standard DMC formulation:

        Y = f + G ΔU

    where:
        Y   : predicted outputs
        f   : free response
        G   : dynamic matrix (Toeplitz-based)
        ΔU  : future control increments

    Parameters
    ----------
    model : DynamicMatrix
            Process model based on step response.

    Attributes
    ----------
    ny : int
         Number of outputs.

    nu : int
         Number of inputs.

    Np : int
         Prediciton horizon.

    Nc : int
         Control horizon.

    G  : ndarray
         Dynamic matrix.
         Shape(ny*Np, nu*Nc)
    
    Notes
    -----
    The model assumes:
    - Linearity (superposition holds)
    - Time invariance
    - Valid step response representation of system dynamics
    - The response have reached the steady-state (settled)

    Examples
    --------
    >>> dyn_matrix = DynamicMatrix(Np=20, Nc=5, y=y, step_amp=step_amp)
    >>> model = StepResponseModel(dyn_matrix)

    """
    
    def __init__(
            self,
            model: DynamicMatrix | None,
            ) -> None:
        
        self.FeedbackState = FeedbackState

        if model is not None:
            self.nu = model.nu
            self.ny = model.ny
            
            self.Np = model.Np
            self.Nc = model.Nc
            
            self.G = model.G
            self.S = model.S

            self.deltaS = model.deltaS
    
    @classmethod
    def load_from_npz(cls, filepath: str):
        """
        Alternative constructor: load model attributes from .npz file and returns the class instances.

        Parameters
        ----------
        filepath : str
                   current path of the .npz model.

        Returns
        -------
        instance : instances of the model
        """

        data = np.load(filepath)

        instance = cls(model=None)

        instance.G = data['G']
        instance.S = data['S']
        instance.deltaS = data['deltaS']
        instance.Np = int(data['Np'])
        instance.Nc = int(data['Nc'])
        instance.nu = int(data['nu'])
        instance.ny = int(data['ny'])

        return instance

    def free_response(
            self,
            feedback: FeedbackState
            ) -> NDArray[np.float64]:
        """
        Compute the free response over the prediction horizon.

        Parameters
        ----------
        feedback    :   FeedbackState
                        Contains the available feedback information for the evaluation of the free response, the measurement of the outputs and the increments history.

        Returns
        -------
        free response   :   NDArray[np.float64]
                            Free response prediction vector, corrected by the measured outputs.
                            Shape(ny, Np)
        """
        return compute_free_response(self.deltaS, feedback)

    def interpolate(self):
        ...

    @property
    def gain(self) -> NDArray[np.float64]:
        return self.S[:, -1].reshape(self.ny, self.nu)