"""
Simulation utilities: rollout with a generic controller callable.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np


def simulate(
    Ad: np.ndarray,
    Bd: np.ndarray,
    controller: Callable[[np.ndarray, int], float],
    x0: np.ndarray,
    steps: int,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate x_{k+1} = Ad x_k + Bd u_k with a user-supplied controller.

    Args:
        Ad, Bd: discrete-time system matrices
        controller: function (x_k, k) -> u_k
        x0: initial state (n,)
        steps: number of time steps
        Q, R: optional cost weights for reporting total quadratic cost

    Returns:
        xs: (steps+1, n) state trajectory
        us: (steps,) control sequence
        total_cost: sum x^T Q x + u^T R u (if Q/R given, else 0.0)
    """
    x = np.asarray(x0, dtype=float).reshape(-1)
    n = x.size
    xs = np.zeros((steps + 1, n), dtype=float)
    us = np.zeros((steps,), dtype=float)

    xs[0] = x
    total_cost = 0.0

    for k in range(steps):
        u = float(controller(xs[k], k))
        us[k] = u
        xs[k + 1] = Ad @ xs[k] + Bd.flatten() * u

        if Q is not None:
            total_cost += xs[k] @ Q @ xs[k]
        if R is not None:
            total_cost += u * (R.reshape(1,1)[0,0] if R.size == 1 else float(np.array([u]) @ R @ np.array([u])))

    return xs, us, total_cost


def lqr_controller(K: np.ndarray) -> Callable[[np.ndarray, int], float]:
    """
    Create a state-feedback controller u = -K x.
    """
    def ctrl(xk: np.ndarray, k: int) -> float:  # noqa: ARG001
        return float(-K @ xk)
    return ctrl
