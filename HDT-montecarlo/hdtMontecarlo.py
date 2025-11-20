# Problema 1: Diagnóstico de cáncer de mama (PPV)
# ------------------------------------------------
# (a) Simula una población de 100,000 mujeres donde el 1% tiene cáncer.
# (b) Aplica la prueba (sens. 90%, FPR 10%) y reporta verdaderos positivos y falsos positivos.
# (c) Usa la simulación para estimar P(cáncer | +) y compárala con Bayes.

from __future__ import annotations
import numpy as np
import math
from dataclasses import dataclass
from collections import Counter


SEED = 2025
rng = np.random.default_rng(SEED)


@dataclass
class CIResult:
    estimate: float
    ci_low: float
    ci_high: float
    n: int

def proportion_ci(p_hat: float, n: int, alpha: float = 0.05) -> CIResult:
    z = 1.96
    se = math.sqrt(max(p_hat * (1 - p_hat), 1e-16) / max(n, 1))
    return CIResult(p_hat, max(0.0, p_hat - z*se), min(1.0, p_hat + z*se), n)


def caminata_100() -> int:

    pasos = rng.choice([-1, 1], size=100)
    pos_final = int(pasos.sum())
    return pos_final


def estimar_prob_retorno(R: int = 10_000) -> CIResult:

    exitos = 0
    for _ in range(R):
        exitos += int(caminata_100() == 0)
    p_hat = exitos / R
    return proportion_ci(p_hat, R)


if __name__ == "__main__":

    pos = caminata_100()
    print("Inciso (a):")
    print(f"  Posición final después de 100 pasos: {pos}")
    print()

    
    R = 10_000
    ci = estimar_prob_retorno(R=R)
    print("Inciso (b):")
    print(f"  Estimación P(X_100 = 0): {ci.estimate:.4f}")
    print(f"  IC 95%: [{ci.ci_low:.4f}, {ci.ci_high:.4f}]  (R = {ci.n})")
