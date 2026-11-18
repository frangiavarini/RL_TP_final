"""Evaluación en un conjunto fijo de laberintos no vistos en entrenamiento.

Ejemplos:
    python evaluate.py runs/A3_goal_s0                       # 100 episodios, marcas normales
    python evaluate.py runs/A3_goal_s0 --mark-view hidden    # test causal: marcas ocultas
    python evaluate.py runs/A3_goal_s0 --mark-view random    # test causal: marcas aleatorias
    python evaluate.py runs/A3_goal_s0 --save-maps 10        # mapas de marcas + trayectoria

Salidas en <run>/eval/<view>_<model>/:
    episodes.csv  una fila por episodio (éxito, pasos, % revisitadas, ...)
    steps.csv     una fila por paso (celda, valor escrito, contexto) para el
                  análisis de P(bit=1 | contexto)
    summary.json  medias y error estándar
    maps/*.png    mapas de marcas al final del episodio
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

import marcas  # noqa: F401

EVAL_SEED_OFFSET = 1_000_000  # train usa seeds < 1e6 (seed*1000 + i)


def load_model(run_dir: Path, model_name: str, recurrent: bool, device: str):
    path = run_dir / model_name
    if recurrent:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, device=device)
    from stable_baselines3 import PPO
    return PPO.load(path, device=device)


def draw_map(env, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from marcas.env import MARK_COLORS

    R, C = env.grid_rows, env.grid_cols
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for (r, c) in env.visited:
        ax.add_patch(Rectangle((c, r), 1, 1, color="#e8e8e8", lw=0))
    for r in range(R):
        for c in range(C):
            v = int(env.marks[r, c])
            if v:
                ax.add_patch(Rectangle((c + 0.15, r + 0.15), 0.7, 0.7,
                                       color=MARK_COLORS[v], lw=0))
    # paredes: entre celdas adyacentes no accesibles + borde
    for r in range(R):
        for c in range(C):
            if c + 1 < C and not env.accessible((r, c), (r, c + 1)):
                ax.plot([c + 1, c + 1], [r, r + 1], "k-", lw=2)
            if r + 1 < R and not env.accessible((r, c), (r + 1, c)):
                ax.plot([c, c + 1], [r + 1, r + 1], "k-", lw=2)
    ax.plot([0, C, C, 0, 0], [0, 0, R, R, 0], "k-", lw=3)
    traj = np.array([(c + 0.5, r + 0.5) for r, c in env.trajectory])
    ax.plot(traj[:, 0], traj[:, 1], "-", color="#1f77b4", lw=1.2, alpha=0.8)
    ax.plot(*traj[0], "o", color="#1f77b4", ms=7, label="inicio")
    gr, gc = env.goal_cell
    ax.plot(gc + 0.5, gr + 0.5, "s", color="red", ms=10, label="objetivo")
    ax.set_xlim(0, C)
    ax.set_ylim(R, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default="model.zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--mark-view", default="normal", choices=["normal", "hidden", "random"])
    ap.add_argument("--deterministic", action="store_true",
                    help="acción más probable (por defecto se muestrea: en un POMDP la "
                         "política óptima sin memoria puede ser estocástica)")
    ap.add_argument("--save-maps", type=int, default=0, help="guardar mapas de los primeros N episodios")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override de kwargs del ambiente (p. ej. num_rows=4) para generalización")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    meta = yaml.safe_load(open(args.run_dir / "run.yaml"))
    kw = dict(meta["env_kwargs"], mark_view=args.mark_view)
    for o in args.set:
        k, v = o.split("=", 1)
        kw[k] = yaml.safe_load(v)
    model = load_model(args.run_dir, args.model, meta["recurrent"], args.device)

    tag = f"{args.mark_view}_{Path(args.model).stem}" + ("_det" if args.deterministic else "")
    if args.set:
        tag += "_" + "_".join(s.replace("=", "") for s in args.set)
    out = args.run_dir / "eval" / tag
    (out / "maps").mkdir(parents=True, exist_ok=True)

    env = gym.make("MarkedMaze-v0", **kw)
    u = env.unwrapped
    episodes, steps_rows = [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=EVAL_SEED_OFFSET + ep)
        lstm_state, start = None, np.ones((1,), dtype=bool)
        d0 = int(u.goal_dist[info["cell"]])
        t = 0
        while True:
            cell = u.pos_to_cell(u.agent.pos)
            ctx = dict(n_open=u.n_open(cell), already_visited=None, already_marked=int(u.marks[cell]))
            if meta["recurrent"]:
                action, lstm_state = model.predict(obs, state=lstm_state, episode_start=start,
                                                   deterministic=args.deterministic)
                start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            visits_before = sum(1 for c in u.trajectory if c == cell)
            obs, r, term, trunc, info = env.step(action)
            steps_rows.append(dict(episode=ep, t=t, row=cell[0], col=cell[1], move=int(action[0]),
                                   value=int(action[1]), n_open=ctx["n_open"],
                                   visits_to_cell=visits_before, already_marked=ctx["already_marked"],
                                   dist_goal=int(u.goal_dist[cell])))
            t += 1
            if term or trunc:
                break
        m = dict(info["episode_metrics"], episode=ep, seed=EVAL_SEED_OFFSET + ep, d0=d0)
        episodes.append(m)
        if ep < args.save_maps:
            draw_map(u, out / "maps" / f"ep{ep:03d}.png",
                     f"{meta['agent']} ({args.mark_view}) ep {ep}: "
                     f"{'éxito' if m['success'] else 'fallo'} en {m['steps']} pasos")

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0]))
        w.writeheader()
        w.writerows(episodes)
    with open(out / "steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(steps_rows[0]))
        w.writeheader()
        w.writerows(steps_rows)

    def stat(key, rows):
        x = np.array([r[key] for r in rows], dtype=float)
        if len(x) == 0:
            return dict(mean=None, sem=None, n=0)
        return dict(mean=float(x.mean()), sem=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
                    n=len(x))

    succ = [e for e in episodes if e["success"]]
    summary = dict(agent=meta["agent"], seed=meta["seed"], mark_view=args.mark_view,
                   model=args.model, deterministic=args.deterministic, env_overrides=args.set,
                   success=stat("success", episodes), steps_success=stat("steps", succ),
                   revisit_frac=stat("revisit_frac", episodes), write_rate=stat("write_rate", episodes),
                   return_base=stat("return_base", episodes))
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps({k: (v["mean"] if isinstance(v, dict) else v) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
