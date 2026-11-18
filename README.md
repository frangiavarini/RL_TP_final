# TP Final RL — Memoria externa escrita en el ambiente

Agente con cámara frontal en un laberinto 3D (MiniWorld `MazeS3Fast`) que, además de moverse,
escribe marcas en el piso. Comparamos cuatro agentes: sin memoria (A0), LSTM (A1),
marca fija (A2) y marca aprendida (A3). Consigna y alcance: [docs/consigna_tp_final.pdf](docs/consigna_tp_final.pdf).

## Estado del trabajo (17/11)

Hecho y probado localmente (issues [#2](../../issues/2), [#3](../../issues/3), [#4](../../issues/4),
[#6](../../issues/6), [#10](../../issues/10)). Pendiente: correr los entrenamientos completos en la VM.

### Qué hay en el repo

| Archivo | Qué hace |
|---|---|
| [docs/consigna_tp_final.pdf](docs/consigna_tp_final.pdf) | Nuestra consigna: alcance acordado con la cátedra, objetivos mínimos, cronograma |
| [consigna/](consigna/) | Consigna general del proyecto final de la cátedra |
| [marcas/env.py](marcas/env.py) | El ambiente `MarkedMaze-v0` (MazeS3Fast + marcas + shaping + métricas) |
| [marcas/agents.py](marcas/agents.py) | Definición de A0–A3 (qué `mark_mode` usa cada uno y si es recurrente) |
| [configs/base.yaml](configs/base.yaml) | Hiperparámetros comunes a los 4 agentes |
| [train.py](train.py) | Entrenamiento PPO / RecurrentPPO con logging de las métricas del TP |
| [evaluate.py](evaluate.py) | Evaluación, test causal de las marcas, mapas de marcas, CSV por paso |
| [scripts/run_all.sh](scripts/run_all.sh) | Entrena y evalúa todo (agentes × semillas) en paralelo; usa xvfb en Linux |
| [scripts/summarize.py](scripts/summarize.py) | Tabla final: media ± desvío entre semillas por agente y tipo de evaluación |
| [tests/test_env.py](tests/test_env.py) | 11 tests del ambiente |

### Decisiones de diseño (y por qué)

- **Ambiente reimplementado desde cero.** El ambiente que ya teníamos no estaba en el repo, así que
  este es una subclase de `miniworld.envs.maze.Maze` con los parámetros de `MazeS3Fast`
  (paso 0.7, giro 45°, 300 pasos). Hay que compararlo con la otra implementación (#1) y quedarnos con una.
- **Celdas de 1×1.** Cada habitación de 3×3 se divide en 9 celdas → grilla de 9×9. Con habitaciones
  enteras como celdas (3×3) las marcas serían demasiado gruesas para codificar recorridos. Como el
  paso es 0.7, el agente pasa ~1–2 pasos por celda.
- **Se escribe antes de moverse**, en la celda donde está el agente al elegir la acción
  ("queda escrito en la posición donde está"). Escribir 0 no borra (la marca es persistente).
- **Las marcas se dibujan** como cuadrados de color sobre el piso (amarillo = 1; con 2 bits también
  celeste y magenta). Se ven en la cámara y en la vista cenital. La cámara no ve el piso justo debajo
  del agente: por eso es útil el vector `marks` que sugirió Gabriel.
- **Vector `marks` egocéntrico.** One-hot de la marca de [actual, adelante, izquierda, derecha, atrás]
  + 4 flags de "vecina accesible" (sin pared). La dirección del agente se redondea a la cardinal más
  cercana (con giros de 45° hay diagonales; en empate se elige una de forma determinística).
- **Mismo espacio de acción y observación para los 4 agentes.** En A0/A1 la cabeza del bit existe
  pero se ignora y el vector `marks` es siempre "sin marca". Así la única diferencia es la memoria.
- **A2 marca la celda inicial y cada celda que visita**; el bit del agente se ignora.
- **Shaping basado en potencial** (Ng et al. 1999) con el **mismo γ que PPO** y Φ(terminal)=0, que es
  lo que garantiza que no cambia la política óptima. Dos potenciales:
  `goal` (−distancia BFS al objetivo / máx.) y `expl` (fracción de celdas visitadas, que no revela
  dónde está el objetivo). Con Φ ≤ 0 el retorno *sin descontar* puede dar > 1 aun fallando: no es
  un bug (la invariancia es del retorno descontado, ver `test_shaping_discounted_telescopes`), pero
  por eso **todas las métricas usan la recompensa original** (`info["reward_base"]`).
- **Truncation vs. termination.** Llegar al objetivo es `terminated`; agotar los 300 pasos es
  `truncated` (SB3 hace bootstrap del valor). El kwarg se llama `max_steps` porque `gym.make`
  intercepta `max_episode_steps`.
- **Métricas** (en `info["episode_metrics"]` al final del episodio): éxito, pasos, retorno original,
  `% revisitadas` = entradas a celdas ya visitadas / total de entradas a celdas, celdas visitadas,
  tasa de escritura, celdas marcadas.
- **Entropía por cabeza** (`tp/entropy_move`, `tp/entropy_bit`) en TensorBoard: si la del bit colapsa
  temprano, A3 deja de explorar qué escribir (candidato a caso de diagnóstico).
- **Evaluación estocástica por defecto.** En la prueba, la política determinista sin memoria se
  quedaba girando en el lugar: en un POMDP la mejor política sin memoria puede ser estocástica
  (también lo vio Cool Antz).
- **Semillas separadas.** Entrenamiento usa `seed*1000 + i`; evaluación usa 1.000.000 + episodio, así
  que los laberintos de evaluación nunca se vieron en entrenamiento.
- **Test causal.** `mark_view=hidden` (el agente escribe pero no ve nada) y `mark_view=random`
  (ve marcas aleatorias). Si A3 no empeora, las marcas se escriben pero no se usan.
- **Reproducibilidad.** Cada corrida guarda `run.yaml` con la config completa, los argumentos y el
  commit (con `-dirty` si había cambios sin commitear): sirve para el caso de diagnóstico.

### Números de referencia (Mac M-series, CPU)

- Agente aleatorio: ~37 % de éxito en 300 pasos (30 laberintos).
- Ambiente solo: ~1760 pasos/s; entrenamiento PPO con 4 envs: ~450 pasos/s (≈3 h por corrida de 5M).
  En la VM hay que medirlo (#5): con xvfb el render puede ser por software.

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
