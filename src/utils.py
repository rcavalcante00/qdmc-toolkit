from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
    
    def to_numpy(self) -> dict:

        fields = self.data[0].__dataclass_fields__.keys()

        return {
            field: np.asarray([getattr(entry, field) for entry in self.data])
            for field in fields
        }
    
    def _reset(self):
        self.data: list[LogEntry] = []
