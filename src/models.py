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
    
class LPVStepResponseModel:
    
    def __init__(
            self,
            models: list[StepResponseModel],
            scheduling_points: NDArray[np.float64],
            rho: NDArray[np.float64] | float
            ):            
       
        if len(models) != len(scheduling_points):
            raise ValueError(
                "The number of models must be equal to the number of scheduling variables."
                )
    
        self.models = models

        self.rhos = np.asarray(scheduling_points, dtype=np.float64)

        self.Np = models[0].Np
        self.Nc = models[0].Nc
        self.ny = models[0].ny
        self.nu = models[0].nu

        self.interpolate(rho)

    def interpolate(
            self,
            rho: NDArray[np.float64] | float
            ):
        
        rho_clipped = np.clip(rho, self.rhos[0], self.rhos[-1])

        idx = np.searchsorted(self.rhos, rho_clipped)

        if idx == len(self.rhos):
            idx -= 1
        idx0 = max(0, idx - 1)
        idx1 = idx

        if idx0 == idx1:
            self.deltaS = self.models[idx0].deltaS
            self.G = self.models[idx0].G
            return
        
        rho0, rho1 = self.rhos[idx0], self.rhos[idx1]
        alpha = (rho_clipped - rho0) / (rho1 - rho0)
        deltaS_interp = (1 - alpha) * self.models[idx0].deltaS + alpha * self.models[idx1].deltaS
        G_interp = (1 - alpha) * self.models[idx0].G + alpha * self.models[idx1].G

        self.deltaS = deltaS_interp
        self.G = G_interp
    
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
    
    # @classmethod
    # def load_from_npz(cls, 
    #                     filepath: str,
    #                     rho: NDArray[np.float64] | float
    #                     ):

    #     models_dict: dict  = np.load(filepath)
        
    #     models = models_dict['models']
    #     rhos = models_dict['rhos']

    #     ref_model = models[0]

    #     instance = cls.__new__(cls)

    #     instance.FeedbackState = FeedbackState

    #     instance.Np = int(ref_model['Np'])
    #     instance.Nc = int(ref_model['Nc'])
    #     instance.nu = int(ref_model['nu'])
    #     instance.ny = int(ref_model['ny'])

    #     instance.rhos = rhos
        
    #     instance.G = self.interpolate(rhos[0])
    #     # instance.S = data['S']
    #     instance.deltaS = model['deltaS']

    #     return instance
        
        # if scheduling is None:
        #     if ext_model is None:
        #         if y is None:
        #             raise ValueError(
        #                 "External model not specified, please provide the step response vector (y)."
        #                 )
        #         if step_amplitudes is None:
        #             raise ValueError(
        #                 "Step response amplitudes must be informed."
        #              )
        #         self.y = self._fit_y_to_horizon(y, Np) if y.shape[1] != Np else np.asarray(y, dtype=np.float64)
                
        #         self.step_amplitudes = np.asarray(step_amplitudes, dtype=np.float64)

        #         self.nu = self.step_amplitudes.size
        #         self.ny = y.shape[0] // self.nu

        #         self.G, self.S = self._build_dynamic_matrix(y=self.y, step_amplitudes=self.step_amplitudes)
            
        #         self.deltaS =  self._build_delta_s(S=self.S)

        # if scheduling is not None:
        #     y0, y1 = scheduling['step0'], scheduling['step1']
        #     amp0, amp1 = scheduling['amplitudes0'], scheduling['amplitudes1']
        #     self.rho0, self.rho1 = scheduling['rho0'], scheduling['rho1']

        #     y0, y1 = [self._fit_y_to_horizon(_, Np) if _.shape[1] != Np else np.asarray(_, dtype=np.float64)
        #               for _ in [y0, y1]]
            
        #     amp0, amp1 = np.asarray(amp0, dtype=np.float64), np.asarray(amp1, dtype=np.float64)

        #     self.nu = amp0.size
        #     self.ny = y0.shape[0] // self.nu

        #     self.G0, self.S0 = self._build_dynamic_matrix(y=y0, step_amplitudes=amp0)
        #     self.G1, self.S1 = self._build_dynamic_matrix(y=y1, step_amplitudes=amp1)

        #     self.G = self.interpolate(rho=u_nom, x0=self.G0, x1=self.G1)
        #     self.S = self.interpolate(rho=u_nom, x0=self.S0, x1=self.S1)

        #     self.deltaS0 =  self._build_delta_s(S=self.S0)
        #     self.deltaS1 =  self._build_delta_s(S=self.S1)
            
        #     self.deltaS = self.interpolate(rho=u_nom, x0=self.deltaS0, x1=self.deltaS1)

    # @property
    # def gain(self) -> NDArray[np.float64]:
    #     return self.S[:, -1].reshape(self.ny, self.nu)
    
class StateSpaceModel:

    def __init__(
            self,
            Np: int,
            Nc: int,
            A: NDArray[np.float64],
            B: NDArray[np.float64],
            C: NDArray[np.float64],
            D: NDArray[np.float64],
            DT: NDArray[np.float64] | float,
            x_nom: NDArray[np.float64],
            u_nom: NDArray[np.float64]
    ):
    
        self.Np = Np
        self.Nc = Nc

        result = cast( # the cast is for typing purposes only
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            float,
        ],
        cont2discrete((A, B, C, D), DT, method="zoh"),
        )

        Ad, Bd, Cd, Dd, _ = result
        self.A_d = Ad
        self.B_d = Bd
        self.C_d = Cd
        
        self.x_nom = x_nom
        self.y_nom = self.C_d @ x_nom
        self.u_nom = u_nom
        self.z_nom = np.hstack([x_nom, u_nom])
        
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]

        self.Ad_aug = np.block([[Ad, Bd],
                                [np.zeros((self.nu, self.nx)), np.eye(self.nu)]])
        
        self.Bd_aug = np.vstack([Bd, np.eye(self.nu)])

        self.Cd_aug = np.hstack([Cd, Dd])

        self.Dd_aug = np.zeros((self.ny, self.nu))
        
        self._build_A_powers()
        self._build_Phi()
        
        self.gain = Cd @ np.linalg.inv(np.eye(self.nx) - Ad) @ Bd + Dd

        self.prediction = np.zeros(self.ny)
    
    def _build_A_powers(self):
        self.A_powers = np.empty((self.Np + 1, self.nx + self.nu, self.nx + self.nu))
        self.A_powers[0] = np.eye(self.nx + self.nu)

        for k in range(1, self.Np + 1):
           self.A_powers[k] = self.A_powers[k-1] @ self.Ad_aug

    def _build_Phi(self):
        
        Np = self.Np
        Nc = self.Nc

        nx = self.nx
        nu = self.nu
        ny = self.ny

        Phi = np.zeros((Np * ny, Nc * nu))

        for i in range(Np):
            for j in range(Nc):
                if i >= j:
                    power = i - j
                    if power == 0:
                        block = self.Cd_aug @ self.Bd_aug
                    else:
                        block = self.Cd_aug @ self.A_powers[power - 1] @ self.Bd_aug

                    row = i * ny
                    col = j * nu
                    Phi[row:row + ny, col:col + nu] = block

        Phi_4d = Phi.reshape(Np, ny, Nc, nu)

        Phi_reordenada = Phi_4d.transpose(1, 0, 3, 2)

        # self.Phi = Phi
        self.Phi = Phi_reordenada.reshape(ny * Np, nu * Nc)
        self.G = self.Phi
    
    # def _free_response(self,
    #                    x_current: NDArray[np.float64],
    #                    u_current: NDArray[np.float64],
    #                    ) -> NDArray[np.float64]:
        
    #     z_current = np.hstack([x_current, u_current])
    #     y_current = self.Cd_aug @ z_current
        
    #     F = np.zeros(self.Np * self.ny)

    #     for i in range(self.Np):
    #         z_pred = self.A_powers[i + 0] @ (z_current - self.z_nom) # voltar aqui depois e colocar i + 1
    #         y_free = self.Cd_aug @ z_pred
    #         F[i * self.ny : (i+1) * self.ny] = y_free
    def _free_response(self,
                        y_meas: NDArray[np.float64],
                        u_past: NDArray[np.float64], # Ação de controle aplicada no instante anterior
                        ) -> NDArray[np.float64]:
            
            # 1. Inicializa o estado interno na primeira execução
            # (Alternativamente, você pode colocar self.x_current = self.x_nom.copy() no __init__)
            if not hasattr(self, 'x_current'):
                self.x_current = self.x_nom.copy()
                
            # 2. Propaga o estado do modelo: x(k|k-1) = A * x(k-1) + B * u(k-1)
            self.x_current = self.A_d @ self.x_current + self.B_d @ u_past
            
            # 3. Monta o estado aumentado atual z(k) para o formato de velocidade
            z_current = np.hstack([self.x_current, u_past])
            
            # 4. Resposta livre do modelo (em variáveis de desvio)
            f_dev = np.zeros(self.Np * self.ny)
            
            for i in range(self.Np):
                z_pred = self.A_powers[i + 1] @ (z_current - self.z_nom) 
                y_free_dev = self.Cd_aug @ z_pred
                f_dev[i * self.ny : (i+1) * self.ny] = y_free_dev
                
            # 5. Saída atual do modelo (absoluta) para cálculo do erro
            # O que o modelo acha que está acontecendo agora no instante k
            y_model_current = self.Cd_aug @ (z_current - self.z_nom) + self.y_nom
            
            # 6. Cálculo do erro de predição (Realidade - Modelo)
            pred_error = y_meas - y_model_current
            self.error = pred_error
            
            # 7. Predição Final Corrigida (Desvio + Nominal + Erro Aditivo)
            F_abs = f_dev + np.tile(self.y_nom + pred_error, self.Np)
            
            # 8. Reorganiza para o formato em blocos (ny, Np)
            return F_abs.reshape(self.Np, self.ny).T

# if __name__ == '__main__':
#     A = np.array([[-70.44082667,   0.        ,  -2.66222414,   0. ],
#         [ 39.57511372, -59.57511372,   0.        ,   0.        ],
#         [102.60332869, 154.81338196, -40.82400616,  30.82851638],
#         [  0.        ,   0.        ,  86.688     , -86.688     ]])

#     B = np.array([
#         [ 4.1  ,  0.],
#         [-1.   ,  0.   ],
#         [ 0.   ,  0.   ],
#         [ 0.   , 86.688]])

#     C = np.array(
#         [[0, 1, 0, 0],
#         [0, 0, 1, 0]])

#     D = np.array([[0, 0],
#                 [0, 0]])

#     Np = 160
#     Nc = 4


#     vdv_step = np.load(r'C:/Users/rafae/OneDrive/Tese/Controle/Studies/VdV/step20.npz')
#     Y = vdv_step['Y']
#     du = vdv_step['du']

#     from collections import deque

#     du_past = deque([np.zeros(2) for _ in range(Np)], maxlen=Np)

#     dyn_matrix = DynamicMatrix(Np=Np, Nc=Nc, y=Y, step_amp=du)

#     fsr = StepResponseModel(dyn_matrix)
#     # fsr = StepResponseModel(y=Y_desbuta, du=du_desbuta, Np=Np, Nc=Nc)

#     x_nom = np.array([  1.03617958,   0.80396398, 139.53788795, 137.53788795])
#     u_nom = np.array([20, -2])

#     ss = StateSpaceModel(Np=Np,
#                     Nc=Nc,
#                     A=A, 
#                     B=B,
#                     C=C,
#                     D=D,
#                     DT = 1.0,
#                     x_nom = x_nom,
#                     u_nom=u_nom)

#     z_k = np.hstack([x_nom, u_nom]) 

#     pass
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     Desbuta_step = np.load(r'C:/Users/rafae/OneDrive/Tese/Controle/Studies/VdV/step20.npz')
#     Y = Desbuta_step['Y']
#     step_amplitudes= Desbuta_step['du']
    
#     # fsr = StepResponseModel(y=Y, step_amplitudes=step_amplitudes, Np=160, Nc=8)
#     # fsr = StepResponseModel(y=Y_desbuta, du=du_desbuta, Np=Np, Nc=Nc)

#     #%%
#     fig, ax = plt.subplots(
#         fsr.ny,
#         fsr.nu,
#         sharex=True,
#         figsize=(fsr.nu * 2.0, fsr.ny * 1.5)
#         )
    
#     fig.subplots_adjust(
#         wspace=0,
#         hspace=0
#     )
    
#     def auto_fontsize(fig, ny, nu,
#                       scale=5,
#                       min_size=4,
#                       max_size=14):
    
#         w, h = fig.get_size_inches()
    
#         size = min(w / nu, h / ny) * scale
    
#         return max(min_size, min(size, max_size))
#     fontsize = auto_fontsize(fig, fsr.ny, fsr.nu)
#     color1 = 'magenta'
    
#     mvtags = [f'$u_{i+1}$' for i in range(6)]
#     cvtags = [f'$y_{i+1}$' for i in range(6)]
    
#     for i, a in enumerate(ax.flatten()):
        
#         row = i // fsr.nu
#         col = i % fsr.nu
    
#         a.plot(fsr.S[i], color=color1, lw=1.0, label=f'{fsr.gain.flatten()[i]:.5f}')
        
#         a.set_xticklabels([])
#         a.set_yticklabels([])
        
#         a.label_outer()
        
#         a.minorticks_on()
#         a.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25)
#         a.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.15)
        
#         if col != 0:
#             a.tick_params(axis='y', which='both',direction='in', colors='k', top=True, left=False)
        
#         if col == -1:
#             a.tick_params(axis='y', which='both',direction='in', colors='k', top=True, right=True)
            
#         # ylabel apenas primeira coluna
#         if col == 0:
#             a.set_ylabel(cvtags[row])
#             a.tick_params(axis='y', which='both',direction='in', colors='k', top=True, right=False)
    
#         # xlabel apenas última linha
#         if row == fsr.ny - 1:
#             a.set_xlabel(mvtags[col])
#             a.tick_params(axis='x', which='both',direction='in', colors='k', top=False, right=False)
        
#         if col != 0:
#             a.spines['left'].set_visible(False)
    
#         if row != fsr.ny - 1:
#             a.spines['bottom'].set_visible(False)
    
#         a.axhline(0, color='black', lw=0.5, alpha=1.0)
        
#         a.legend(loc=5,
#                  frameon=False,
#                  handlelength=0,
#                  handletextpad=0,
#                  labelcolor=color1,
#                  prop={
#                     'size': fontsize,
#                     'weight': 'bold',
#                        },
#                  )
        
#     plt.show()
        
