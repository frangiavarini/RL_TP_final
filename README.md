# TP Final RL — Memoria externa escrita en el ambiente

Agente con cámara frontal en un laberinto 3D (MiniWorld `MazeS3Fast`) que, además de moverse,
escribe marcas en el piso. Comparamos cuatro agentes: sin memoria (A0), LSTM (A1),
marca fija (A2) y marca aprendida (A3). Consigna y alcance: [docs/consigna_tp_final.pdf](docs/consigna_tp_final.pdf).

## Instalación

```bash
uv venv --python 3.11 .venv && uv pip install -r requirements.txt   # o: python -m venv + pip
export PYTHONPATH=.            # ojo: si tenés un PYTHONPATH de ROS/conda, vacialo antes
pytest -q tests                # 11 tests del ambiente
```

En Linux sin pantalla (VM): `sudo apt install xvfb` y anteponer `xvfb-run -a` a los comandos.

## Ambiente (`marcas/env.py`, id `MarkedMaze-v0`)

| | |
|---|---|
| Grilla | cada habitación 3×3 se divide en celdas 1×1 → 9×9 celdas |
| Acción | `MultiDiscrete([3, 2**n_bits])` = (girar izq / girar der / avanzar, valor). Valor > 0 escribe en la celda actual **antes** de moverse; 0 no borra |
| Observación | `image` (60×80×3) + `marks`: one-hot de la marca en [actual, adelante, izq, der, atrás] + 4 flags de vecina accesible (marco egocéntrico) |
| `mark_mode` | `none` (A0/A1), `fixed` (A2: el ambiente marca cada celda visitada), `learned` (A3) |
| `shaping` | `none`, `goal` (Φ = −d_geo/d_max), `expl` (Φ = celdas visitadas / total). F = γΦ(s′) − Φ(s), Φ(terminal)=0 |
| `mark_view` | `normal`, `hidden` (se escriben pero no se ven), `random` (marcas aleatorias) — para el test causal |
| `info` | `reward_base` (recompensa original), `reward_shaping`, y al final `episode_metrics` |

Nota: el shaping preserva la política óptima para el retorno **descontado**
(Σγᵗ F_t = γᵀΦ(s_T) − Φ(s_0), ver test). El retorno *sin descontar* con shaping puede ser
> 1 en episodios fallidos; por eso todas las métricas usan `reward_base`.

## Entrenar

```bash
python train.py --agent A3 --seed 0                               # config: configs/base.yaml
python train.py --agent A3 --seed 0 --set env.shaping=none        # ablación del shaping
python train.py --agent A0 --seed 0 --set total_timesteps=2e4 --set n_envs=2   # smoke test
tensorboard --logdir runs
```

Cada corrida queda en `runs/<agente>_<shaping>_s<seed>/` con `run.yaml` (config completa +
commit), TensorBoard (`tp/success`, `tp/revisit_frac`, `tp/write_rate`, `tp/entropy_move`,
`tp/entropy_bit`, ...), checkpoints y `model.zip`.

## Evaluar

```bash
python evaluate.py runs/A3_goal_s0 --save-maps 10        # 100 laberintos no vistos
python evaluate.py runs/A3_goal_s0 --mark-view hidden    # test causal
python evaluate.py runs/A3_goal_s0 --mark-view random
python evaluate.py runs/A3_goal_s0 --set num_rows=4 --set num_cols=4   # generalización
python scripts/summarize.py runs                         # tabla media ± desvío entre semillas
```

Por defecto la evaluación **muestrea** acciones (`--deterministic` para la más probable): un
agente sin memoria en un POMDP puede necesitar una política estocástica, y en modo determinista
suele quedarse girando en el lugar.

Salidas en `runs/<run>/eval/<vista>_<modelo>/`: `episodes.csv`, `steps.csv` (un paso por fila
con celda, valor escrito, `n_open` = 1 callejón / 2 pasillo / 3+ cruce, visitas previas, etc.,
para estimar P(bit=1 | contexto)), `summary.json` y `maps/*.png`.

## Todo junto (VM)

```bash
SEEDS="0 1 2" PARALLEL=4 bash scripts/run_all.sh
```
