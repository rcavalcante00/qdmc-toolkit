from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray, ArrayLike
from scipy.signal import max_len_seq

@dataclass(slots=True)
class LogEntry:
    k: int

    y_sp: NDArray[np.float64]
    y_meas: NDArray[np.float64]
    y_pred: NDArray[np.float64]

    e: NDArray[np.float64]
    e_pred: NDArray[np.float64]

    u_meas: NDArray[np.float64]
    move: NDArray[np.float64]
    cost: NDArray[np.float64]

    lbu: NDArray[np.float64] | None = None
    ubu: NDArray[np.float64] | None = None
    lby: NDArray[np.float64] | None = None
    uby: NDArray[np.float64] | None = None

class Logger:

    def __init__(self):
        self.data: list[LogEntry] = []

    @staticmethod
    def _convert(x):

        if x is None:
            return np.asarray(np.nan)
        
        return np.asarray(x, dtype=np.float64).copy()

    def write(
        self,
        y_sp: NDArray[np.float64],
        y_meas: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        u_meas: NDArray[np.float64],
        move: NDArray[np.float64],
        cost: NDArray[np.float64]| float,
        lbu: NDArray[np.float64] | None = None,
        ubu: NDArray[np.float64] | None = None,
        lby: NDArray[np.float64] | None = None,
        uby: NDArray[np.float64] | None = None,
    ):

        entry = LogEntry(
            k=len(self.data),

            y_sp=self._convert(y_sp),
            y_meas=self._convert(y_meas),
            y_pred=self._convert(y_pred),

            e=self._convert(y_sp) - self._convert(y_meas),
            e_pred=self._convert(y_meas) - self._convert(y_pred),

            u_meas=self._convert(u_meas),
            move=self._convert(move),
            cost=self._convert(cost),
            
            lbu=self._convert(lbu),
            ubu=self._convert(ubu),
            lby=self._convert(lby),
            uby=self._convert(uby),
        )

        self.data.append(entry)

    def to_dict(self):
        return [asdict(entry) for entry in self.data]

    def to_df(self) -> pd.DataFrame:

        rows = []

        vector_fields = {
            "y_sp",
            "y_meas",
            "y_pred",
            "e",
            "e_pred",
            "u_meas",
            "move",
            "lbu",
            "ubu",
            "lby",
            "uby",
        }

        for entry in self.data:

            row = {
                "k": entry.k,
                "cost": entry.cost,
            }

            for name in vector_fields:

                value = getattr(entry, name)

                if value is None:
                    continue

                value = np.asarray(value).flatten()

                for i, v in enumerate(value, start=1):
                    row[f"{name}_{i}"] = v

            rows.append(row)

        return pd.DataFrame(rows)
    
    def to_numpy(self) -> Dict:

        fields = self.data[0].__dataclass_fields__.keys()

        return {
            field: np.asarray([getattr(entry, field) for entry in self.data])
            for field in fields
        }
    
    def _reset(self):
        self.data: list[LogEntry] = []

def prbs_mimo(
    nu: int,
    dt: float,
    T_total: Optional[float] = None,
    Ns: Optional[int] = None,
    u0: float | np.ndarray = 0.0,
    a: float | np.ndarray = 1.0,
    T_hold: Optional[float] = None,
    N_model: int = 40,
    method: Literal['independent', 'shifted'] = 'independent',
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera sinal PRBS multivariável para experimentos de identificação.

    Parâmetros
    ----------
    nu : int
        Número de entradas (canais).
    dt : float
        Período de amostragem (em unidades de tempo).
    T_total : float, opcional
        Duração total do experimento. Se fornecida, Ns é calculado.
    Ns : int, opcional
        Número total de amostras (alternativa a T_total).
    u0 : float ou array (nu,)
        Valores nominais das entradas (ponto de operação).
    a : float ou array (nu,)
        Semi-amplitude das excursões (±a em torno de u0).
    T_hold : float, opcional
        Tempo de permanência em cada nível. Se None, T_hold = dt.
    N_model : int
        Ordem estimada do modelo FIR (usada para dimensionar o
        comprimento mínimo da sequência).
    method : {'independent', 'shifted'}
        Método para descorrelacionar os canais:
        - 'independent': mesmo polinômio, estados iniciais diferentes.
        - 'shifted': uma única m‑sequência deslocada circularmente.
    seed : int, opcional
        Semente para reprodutibilidade.

    Retorna
    -------
    t : ndarray (Ns,)
        Vetor de instantes de tempo.
    u : ndarray (Ns, nu)
        Sinais PRBS gerados.
    """

    # Determinar número de amostras
    if T_total is not None:
        Ns = int(np.ceil(T_total / dt))
    elif Ns is None:
        raise ValueError("Forneça T_total ou Ns.")
    else:
        Ns = int(Ns)

    # Tempo de hold em passos de amostragem
    if T_hold is None:
        T_hold = dt
    hold_steps = max(1, int(np.round(T_hold / dt)))

    # Normalizar u0 e a para arrays
    u0 = np.atleast_1d(u0).astype(float)
    a = np.atleast_1d(a).astype(float)
    if u0.size == 1:
        u0 = np.full(nu, u0)
    if a.size == 1:
        a = np.full(nu, a)
    if u0.shape[0] != nu or a.shape[0] != nu:
        raise ValueError("u0 e a devem ter comprimento igual a nu")

    # Comprimento mínimo da sequência PRBS
    L_min = 4 * nu * N_model
    nbits = int(np.ceil(np.log2(L_min + 1)))
    nbits = max(nbits, 2)
    L = 2**nbits - 1  # período da m-sequência

    # Gerador para estados iniciais (se seed fornecido)
    rng = np.random.RandomState(seed)

    # Gerar bits (0/1) para cada canal
    bits = np.zeros((L, nu), dtype=int)

    if method == 'shifted':
        # Uma única sequência, deslocada para cada canal
        seq, _ = max_len_seq(nbits, state=rng.randint(2, size=nbits))
        offset_base = L // nu
        for j in range(nu):
            offset = j * offset_base
            bits[:, j] = np.roll(seq, -offset)
    else:  # independent
        for j in range(nu):
            # Estado inicial não nulo
            state = rng.randint(2, size=nbits)
            if np.all(state == 0):
                state[0] = 1
            seq, _ = max_len_seq(nbits, state=state)
            bits[:, j] = seq

    # Sample & hold: repetir cada bit hold_steps vezes
    bits_hold = np.repeat(bits, hold_steps, axis=0)  # (L*hold_steps, nu)

    # Repetir até atingir Ns amostras
    reps = int(np.ceil(Ns / bits_hold.shape[0]))
    bits_total = np.tile(bits_hold, (reps, 1))[:Ns, :]  # trunca se necessário

    # Mapear para amplitudes físicas: 0 → -a, 1 → +a
    u = u0 + a * (2 * bits_total - 1)

    # Vetor de tempo
    t = np.arange(Ns) * dt

    return t, u

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

def estimate_fir_mimo(U, Y, N, dt=1.0, detrend=True, ridge=0.0):
    # Converte para arrays
    U = np.asarray(U, dtype=float)
    Y = np.asarray(Y, dtype=float)
    
    nsamples, nu = U.shape
    nsamples_y, ny = Y.shape
    
    if nsamples != nsamples_y:
        raise ValueError("U e Y devem ter o mesmo número de amostras")
    if nsamples <= N:
        raise ValueError(f"nsamples ({nsamples}) deve ser maior que N ({N})")
    
    M = nsamples - N   # número de equações

    if detrend:
        u0 = U[0, :].copy()
        y0 = Y[0, :].copy()
        dU = U - u0
        dY = Y - y0
    else:
        dU = U
        dY = Y

    # Índices das amostras atrasadas
    idx = np.arange(N)[::-1]  # [N-1, ..., 0] para pegar a ordem correta
    
    Phi = np.zeros((M, nu * N))

    for j in range(nu):
        col_start = j * N
        col_end = col_start + N
        # Para cada linha k (0..M-1), a saída corresponde ao instante t = k+N
        # As entradas atrasadas são: U[t-1, j], U[t-2, j], ..., U[t-N, j]
        # que equivalem a U[k+N-1, j], ..., U[k, j]
        for d in range(N):
            atraso = N - 1 - d
            Phi[:, col_start + d] = dU[atraso : atraso + M, j]

    H = np.zeros((ny, nu, N))

    for i in range(ny):
        # Sistema: Phi @ theta_i = dY[N:, i]
        if ridge > 0:
            # Regularização de Tikhonov
            A = Phi.T @ Phi + ridge * np.eye(nu*N)
            b = Phi.T @ dY[N:, i]
            theta_i = np.linalg.solve(A, b)
        else:
            theta_i, _, _, _ = np.linalg.lstsq(Phi, dY[N:, i], rcond=None)
        
        # Reorganizar theta_i em H[i, :, :]
        for j in range(nu):
            H[i, j, :] = theta_i[j*N : (j+1)*N]

    S = np.cumsum(H, axis=2)   # integração ao longo do eixo dos atrasos

    return H, S