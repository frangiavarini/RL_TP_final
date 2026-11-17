import math

import gymnasium as gym
import numpy as np
import pytest

import marcas  # noqa: F401
from marcas.env import CARDINALS


def make(**kw):
    return gym.make("MarkedMaze-v0", **kw).unwrapped


def test_spaces():
    e = make(mark_mode="learned", n_bits=2)
    assert e.action_space.nvec.tolist() == [3, 4]
    o, _ = e.reset(seed=0)
    assert o["image"].shape == (60, 80, 3)
    assert o["marks"].shape == (5 * 4 + 4,)


def test_cell_mapping_roundtrip():
    e = make()
    e.reset(seed=0)
    for r in range(e.grid_rows):
        for c in range(e.grid_cols):
            x, z = e.cell_center((r, c))
            assert e.pos_to_cell(np.array([x, 0, z])) == (r, c)


def test_bfs_reaches_all_cells():
    e = make()
    for s in range(5):
        e.reset(seed=s)
        assert (e.goal_dist >= 0).all(), "el laberinto es conexo"


def test_egocentric_left_matches_turn_left():
    """Lo que el vector llama 'izquierda' es adonde mira tras girar 90° a la izq."""
    e = make()
    e.reset(seed=0)
    for k, d in enumerate([0, -math.pi / 2, math.pi, math.pi / 2]):
        e.agent.dir = d
        assert e.heading_index() == k
        c = e.pos_to_cell(e.agent.pos)
        front, left, right, back = e.egocentric_neighbors(c)
        e.agent.dir = d + math.pi / 2  # girar izquierda = +dir en MiniWorld
        assert e.egocentric_neighbors(c)[0] == left
        e.agent.dir = d - math.pi / 2
        assert e.egocentric_neighbors(c)[0] == right


def test_write_semantics():
    e = make(mark_mode="learned")
    e.reset(seed=0)
    c = e.pos_to_cell(e.agent.pos)
    e.step([0, 1])  # girar no cambia de celda
    assert e.marks[c] == 1
    e.step([0, 0])  # bit 0 no borra
    assert e.marks[c] == 1
    o, *_ = e.step([0, 0])
    assert o["marks"][1] == 1.0  # one-hot "celda actual marcada"


def test_none_mode_ignores_bit():
    e = make(mark_mode="none")
    o, _ = e.reset(seed=0)
    for _ in range(5):
        o, *_ = e.step([2, 1])
    assert e.marks.sum() == 0
    assert o["marks"][: 5 * 2 : 2].all()  # todas las celdas "valor 0"


def test_fixed_mode_marks_visited():
    e = make(mark_mode="fixed")
    e.reset(seed=3)
    for _ in range(30):
        _, _, te, tr, _ = e.step([np.random.randint(3), 0])
        if te or tr:
            break
    assert set(zip(*np.nonzero(e.marks))) == e.visited


def test_hidden_view_hides_but_records():
    e = make(mark_mode="learned", mark_view="hidden")
    e.reset(seed=0)
    o, *_ = e.step([0, 1])
    assert e.marks.sum() == 1
    assert o["marks"][0] == 1.0  # se observa "sin marca"


@pytest.mark.parametrize("shaping", ["goal", "expl"])
def test_shaping_discounted_telescopes(shaping):
    """sum_t gamma^t F_t == gamma^T Phi(s_T) - Phi(s_0) (Phi(terminal) = 0)."""
    g = 0.99
    e = make(mark_mode="none", shaping=shaping, gamma=g)
    rng = np.random.default_rng(0)
    for s in range(3):
        e.reset(seed=s)
        phi0 = e._phi()
        tot, t = 0.0, 0
        while True:
            _, _, te, tr, info = e.step([rng.integers(3), 0])
            tot += g ** t * info["reward_shaping"]
            t += 1
            if te or tr:
                phiT = 0.0 if te else e._phi()
                break
        assert tot == pytest.approx(g ** t * phiT - phi0, abs=1e-9)


def test_metrics_present_at_end():
    e = make(mark_mode="fixed", max_steps=20)
    e.reset(seed=0)
    for _ in range(20):
        _, _, te, tr, info = e.step([2, 0])
        if te or tr:
            break
    m = info["episode_metrics"]
    assert set(m) >= {"success", "steps", "revisit_frac", "write_rate"}
