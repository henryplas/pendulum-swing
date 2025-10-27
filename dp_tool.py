# dp_tool.py
# Mesh-based value iteration for swing-up. 2D state grid over (theta, dtheta).
# Uses consistent A->C diagonal per cell for stable barycentric interpolation.

from __future__ import annotations
import numpy as np
from typing import Tuple, Callable

def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))

def _barycentric_in_triangle(p, v0, v1, v2):
    """
    p, v0, v1, v2 : 2D points (tuples/np.array)
    returns lambdas (l0,l1,l2) s.t. p = l0*v0 + l1*v1 + l2*v2, sum=1
    """
    p = np.asarray(p, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    den = ( (v1[1]-v2[1])*(v0[0]-v2[0]) + (v2[0]-v1[0])*(v0[1]-v2[1]) )
    l0 = ( (v1[1]-v2[1])*(p[0]-v2[0]) + (v2[0]-v1[0])*(p[1]-v2[1]) ) / den
    l1 = ( (v2[1]-v0[1])*(p[0]-v2[0]) + (v0[0]-v2[0])*(p[1]-v2[1]) ) / den
    l2 = 1.0 - l0 - l1
    return l0, l1, l2

def _interp_cell_barycentric(theta, dtheta, V, th_grid, dth_grid, i, j):
    """
    Interpolate V at (theta, dtheta) inside cell with corners:
        A = (th[i], dth[j])   B = (th[i+1], dth[j])
        C = (th[i+1], dth[j+1]) D = (th[i], dth[j+1])
    Diagonal is A->C. Choose triangle and sum vertex values with barycentric weights.
    """
    th = th_grid; dv = dth_grid
    A = (th[i],   dv[j])
    B = (th[i+1], dv[j])
    C = (th[i+1], dv[j+1])
    D = (th[i],   dv[j+1])
    P = (theta, dtheta)

    # Side test relative to diagonal A->C
    s = (C[0]-A[0])*(P[1]-A[1]) - (C[1]-A[1])*(P[0]-A[0])
    if s < 0.0:
        # Triangle ABC
        lA, lB, lC = _barycentric_in_triangle(P, A, B, C)
        return lA*V[i, j] + lB*V[i+1, j] + lC*V[i+1, j+1]
    else:
        # Triangle ACD
        lA, lC, lD = _barycentric_in_triangle(P, A, C, D)
        return lA*V[i, j] + lC*V[i+1, j+1] + lD*V[i, j+1]

def interp_barycentric(theta, dtheta, V, th_grid, dth_grid):
    """
    Periodic in theta. Clamps in dtheta.
    """
    # wrap theta into grid period
    th0, th1 = th_grid[0], th_grid[-1]
    width = th1 - th0
    # Map theta to [th0, th1)
    t = th0 + ( (theta - th0) % width )

    # indices
    i = np.searchsorted(th_grid, t) - 1
    j = np.searchsorted(dth_grid, dtheta) - 1
    i = np.clip(i, 0, len(th_grid)-2)
    j = np.clip(j, 0, len(dth_grid)-2)

    # Clamp theta to cell bounds numerically
    tL, tR = th_grid[i], th_grid[i+1]
    dL, dU = dth_grid[j], dth_grid[j+1]
    t = np.clip(t, tL, tR)
    d = float(np.clip(dtheta, dL, dU))

    return _interp_cell_barycentric(t, d, V, th_grid, dth_grid, i, j)

def value_iteration(
    step_fn: Callable[[np.ndarray, float, float], np.ndarray],
    stage_cost: Callable[[np.ndarray, float], float],
    th_grid: np.ndarray,
    dth_grid: np.ndarray,
    actions: np.ndarray,
    dt: float = 0.02,
    gamma: float = 0.995,
    max_iters: int = 300,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      V  : value grid shape (N_th, N_dth)
      Pi : greedy action grid shape (N_th, N_dth)
    """
    V = np.zeros((len(th_grid), len(dth_grid)), dtype=float)
    Pi = np.zeros_like(V)

    for it in range(max_iters):
        V_new = np.empty_like(V)
        delta = 0.0
        for i, th in enumerate(th_grid):
            for j, dth in enumerate(dth_grid):
                x = np.array([th, dth], dtype=float)
                best = np.inf
                best_u = 0.0
                for u in actions:
                    x_next = step_fn(x, float(u), dt)
                    c = stage_cost(x, float(u)) * dt
                    vnext = interp_barycentric(x_next[0], x_next[1], V, th_grid, dth_grid)
                    q = c + gamma * vnext
                    if q < best:
                        best = q
                        best_u = u
                V_new[i, j] = best
                Pi[i, j] = best_u
        delta = np.max(np.abs(V_new - V))
        V = V_new
        # simple stopping
        if delta < tol:
            break
    return V, Pi

def greedy_from_grid(x: np.ndarray, Pi: np.ndarray, th_grid: np.ndarray, dth_grid: np.ndarray) -> float:
    """Nearest-neighbor pick from the discrete policy grid."""
    i = np.searchsorted(th_grid, x[0]) - 1
    j = np.searchsorted(dth_grid, x[1]) - 1
    i = np.clip(i, 0, len(th_grid)-2)
    j = np.clip(j, 0, len(dth_grid)-2)
    return float(Pi[i, j])
