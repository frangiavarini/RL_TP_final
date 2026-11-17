#!/usr/bin/env bash
# Entrena A0–A3 x semillas y evalúa (normal + test causal). Pensado para la VM.
# Uso: SEEDS="0 1 2" PARALLEL=4 bash scripts/run_all.sh [--set total_timesteps=5e6 ...]
set -uo pipefail
SEEDS=${SEEDS:-"0 1 2"}
PARALLEL=${PARALLEL:-4}
AGENTS=${AGENTS:-"A0 A1 A2 A3"}
export PYTHONPATH=.
PY=(python)
# MiniWorld necesita un contexto OpenGL: en Linux sin pantalla usamos xvfb.
if [[ "$(uname)" == "Linux" && -z "${DISPLAY:-}" ]]; then
  PY=(xvfb-run -a -s "-screen 0 1024x768x24" python)
fi
mkdir -p logs

train() {  # agente semilla
  if "${PY[@]}" train.py --agent "$1" --seed "$2" "${@:3}" > "logs/train_$1_s$2.log" 2>&1; then
    echo "OK   train $1 s$2"
  else
    echo "FAIL train $1 s$2 (ver logs/train_$1_s$2.log)"
  fi
}

for s in $SEEDS; do
  for a in $AGENTS; do
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 5; done
    train "$a" "$s" "$@" &
  done
done
wait

for d in runs/*/; do
  [[ -f "$d/model.zip" ]] || continue
  "${PY[@]}" evaluate.py "$d" --save-maps 10 > /dev/null 2>&1 || echo "FAIL eval $d"
  if grep -qE "mark_mode: (fixed|learned)" "$d/run.yaml"; then
    for v in hidden random; do
      "${PY[@]}" evaluate.py "$d" --mark-view $v > /dev/null 2>&1 || echo "FAIL eval $v $d"
    done
  fi
done
python scripts/summarize.py runs | tee results.md
