# QDMC Toolkit — Material da Aula

Implementação didática de um controlador **QDMC (Quadratic Dynamic Matrix Control)**
baseado em modelos de resposta ao degrau, com suporte a restrições de entrada,
saída e variáveis de folga (soft constraints).

## Estrutura do repositório

```
.
├── notebook.ipynb          # Notebook principal da aula
├── requirements.txt
├── .gitignore
├── README.md
└── src/                   # Pacote com o código-fonte
    ├── __init__.py
    ├── models.py            # Modelos de processo (StepResponseModel, DynamicMatrix, FeedbackState)
    ├── optimize.py           # Montagem do problema QP (CostBuilder, ConstraintsBuilder, QPSolve)
    ├── controllers.py         # Controlador QDMC
    ├── utils.py               # Logger de execução e geração de sinais PRBS para identificação
    └── MIMOsys.py              # Simulador de planta (FOPDT/MIMO) para testes, ex: Wood-Berry
```


1. Importe normalmente os módulos do pacote:

```python
from src.controllers import QDMC
from src.models import StepResponseModel, DynamicMatrix
from src.utils import Logger
from src.MIMOsys import Wood_Berry
```

## Dependências

- `numpy`, `scipy`, `pandas` — álgebra linear, sinais e manipulação de dados
- `qpsolvers` — interface unificada para solvers de QP
- `quadprog` — solver QP usado por padrão no controlador (`solver='quadprog'`)
Obs: qualquer solver suportado pelo pacote `qpsolvers` pode ser utilizado.

## Módulos

- **`models.py`** — define o modelo de resposta ao degrau (`StepResponseModel`),
  a matriz dinâmica (`DynamicMatrix`) e o estado de realimentação (`FeedbackState`)
  usados para calcular a resposta livre do sistema.
- **`optimize.py`** — monta a Hessiana, o vetor de custo linear e as restrições
  do problema de programação quadrática, além de resolver o QP (`QPSolve`).
- **`controllers.py`** — implementa o controlador `QDMC`, que integra modelo,
  custo e restrições a cada instante de amostragem.
- **`utils.py`** — `Logger` para registrar a execução do controlador e
  `prbs_mimo`/funções de identificação FIR para gerar dados de teste.
- **`MIMOsys.py`** — simulador de plantas FOPDT (inclui o clássico exemplo
  Wood-Berry), útil para testar o controlador em malha fechada sem hardware real.
