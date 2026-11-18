"""Las cuatro variantes de agente del TP (ver docs/consigna_tp_final.pdf, sec. 4).

Todas comparten el mismo espacio de acción (MultiDiscrete [3, 2**n_bits]) y la
misma observación; solo cambia la fuente de memoria. En A0/A1 la cabeza del
bit existe pero el ambiente la ignora.
"""
AGENTS = {
    "A0": dict(mark_mode="none", recurrent=False, desc="sin memoria"),
    "A1": dict(mark_mode="none", recurrent=True, desc="recurrente (LSTM)"),
    "A2": dict(mark_mode="fixed", recurrent=False, desc="marca fija"),
    "A3": dict(mark_mode="learned", recurrent=False, desc="marca aprendida"),
}
