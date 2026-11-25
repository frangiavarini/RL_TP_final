"""Entrenamiento de los agentes A0–A3 (ver docs/consigna_tp_final.pdf).

Ejemplos:
    python train.py --agent A3 --seed 0
    python train.py --agent A1 --seed 2 --set env.shaping=none --set total_timesteps=2e6
    python train.py --agent A0 --seed 0 --set total_timesteps=20000 --set n_envs=2   # smoke test
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import marcas  # noqa: F401  (registra MarkedMaze-v0)
from marcas.agents import AGENTS


def parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for o in overrides:
        key, val = o.split("=", 1)
        d = cfg
        *path, last = key.split(".")
        for p in path:
            d = d.setdefault(p, {})
        v = parse_value(val)
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                pass
        d[last] = v
    return cfg


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def env_kwargs(cfg: dict, agent: str, **extra) -> dict:
    kw = dict(cfg["env"])
    kw["mark_mode"] = AGENTS[agent]["mark_mode"]
    kw["gamma"] = cfg["ppo"]["gamma"]
    kw.update(extra)
    return kw


def make_env_fn(kwargs: dict, seed: int):
    def _init():
        import marcas  # noqa: F401  (necesario con start_method=spawn)
        env = gym.make("MarkedMaze-v0", **kwargs)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(cfg, agent, seed, n_envs):
    kw = env_kwargs(cfg, agent)
    fns = [make_env_fn(kw, seed * 1000 + i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(fns)
    return VecMonitor(venv)


class EpisodeMetricsCallback(BaseCallback):
    """Loguea las métricas del TP (con la recompensa original) y la entropía por cabeza."""

    def __init__(self):
        super().__init__()
        self.buf = defaultdict(list)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            m = info.get("episode_metrics")
            if m is None:
                continue
            for k, v in m.items():
                self.buf[k].append(v)
            if m["success"]:
                self.buf["steps_success"].append(m["steps"])
        return True

    def _on_rollout_end(self) -> None:
        for k, v in self.buf.items():
            if v:
                self.logger.record(f"tp/{k}", float(np.mean(v)))
        self.buf.clear()
        # Entropía por cabeza de acción (movimiento / bit). Solo PPO no recurrente.
        try:
            rb = self.model.rollout_buffer
            obs = {k: torch.as_tensor(v[:, :8].reshape(-1, *v.shape[2:]), device=self.model.device)
                   for k, v in rb.observations.items()}
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs)
            for name, d in zip(["move", "bit"], dist.distribution):
                self.logger.record(f"tp/entropy_{name}", float(d.entropy().mean()))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=list(AGENTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override de la config, p. ej. env.shaping=none")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume-finished", action="store_true",
                    help="reentrenar aunque ya exista model.zip")
    args = ap.parse_args()

    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.set)
    cfg["total_timesteps"] = int(cfg["total_timesteps"])
    cfg["n_envs"] = int(cfg["n_envs"])
    agent = AGENTS[args.agent]
    name = args.name or f"{args.agent}_{cfg['env']['shaping']}_s{args.seed}"
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(agent=args.agent, agent_desc=agent["desc"], seed=args.seed,
                recurrent=agent["recurrent"], commit=git_commit(),
                argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                env_kwargs=env_kwargs(cfg, args.agent), config=cfg)
    yaml.safe_dump(meta, open(run_dir / "run.yaml", "w"), sort_keys=False, allow_unicode=True)

    venv = make_vec_env(cfg, args.agent, args.seed, cfg["n_envs"])
    ppo_kw = {k: (float(v) if isinstance(v, str) else v) for k, v in cfg["ppo"].items()}
    common = dict(env=venv, seed=args.seed, verbose=1, device=args.device,
                  tensorboard_log=str(run_dir / "tb"), **ppo_kw)
    if agent["recurrent"]:
        from sb3_contrib import RecurrentPPO
        Algo, policy, extra = RecurrentPPO, "MultiInputLstmPolicy", dict(policy_kwargs=dict(cfg["recurrent"]))
    else:
        Algo, policy, extra = PPO, "MultiInputPolicy", {}

    # Reanudar desde el último checkpoint (p. ej. si una VM spot se apagó).
    ckpts = sorted((run_dir / "checkpoints").glob("model_*_steps.zip"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if (run_dir / "model.zip").exists() and not args.resume_finished:
        print(f"Ya terminado: {run_dir}/model.zip (usar --resume-finished para reentrenar)")
        venv.close()
        return
    if ckpts:
        print(f"Reanudando desde {ckpts[-1]}")
        model = Algo.load(ckpts[-1], env=venv, device=args.device,
                          tensorboard_log=str(run_dir / "tb"))
        remaining = cfg["total_timesteps"] - model.num_timesteps
        reset = False
    else:
        model = Algo(policy, **extra, **common)
        remaining, reset = cfg["total_timesteps"], True

    ckpt = CheckpointCallback(save_freq=max(cfg["checkpoint_every"] // cfg["n_envs"], 1),
                              save_path=str(run_dir / "checkpoints"), name_prefix="model")
    t0 = time.time()
    model.learn(max(remaining, 0), callback=CallbackList([EpisodeMetricsCallback(), ckpt]),
                tb_log_name="run", reset_num_timesteps=reset)
    model.save(run_dir / "model.zip")
    dt = time.time() - t0
    json.dump(dict(seconds=dt, fps=cfg["total_timesteps"] / dt),
              open(run_dir / "train_time.json", "w"), indent=2)
    print(f"Listo: {run_dir} ({dt / 60:.1f} min, {cfg['total_timesteps'] / dt:.0f} fps)")
    venv.close()


if __name__ == "__main__":
    main()
