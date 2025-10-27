"""
LQR implementations for discrete-time systems.

We avoid external control toolboxes and implement:
- Infinite-horizon DLQR via fixed-point iteration of the DARE
- Finite-horizon LQR via backward recursion
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np


def dare_iterate(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 tol: float = 1e-10, max_iters: int = 100000) -> np.ndarray:
    """
    Solve the Discrete Algebraic Riccati Equation (DARE) by fixed-point iteration.
    This converges for stabilizable/detectable (A,B,Q^(1/2)) in typical LQR settings.

    P_{k+1} = Q + A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A

    Args:
        A, B: system matrices
        Q, R: cost matrices (Q >= 0, R > 0)
        tol: convergence tolerance on ||P_{k+1} - P_k||_inf
        max_iters: iteration cap

    Returns:
        P: stabilizing solution to DARE
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    P = Q.copy()
    AT = A.T
    BT = B.T

    for _ in range(max_iters):
        S = R + BT @ P @ B
        K = np.linalg.solve(S, BT @ P @ A)  # (R + B^T P B)^{-1} B^T P A
        P_next = Q + AT @ P @ A - AT @ P @ B @ K
        if np.max(np.abs(P_next - P)) < tol:
            return P_next
        P = P_next

    raise RuntimeError("DARE iteration did not converge. Try adjusting Q/R or check (A,B).")


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infinite-horizon discrete-time LQR gain.

    Returns:
        K: optimal feedback gain (u = -K x)
        P: Riccati solution
    """
    P = dare_iterate(A, B, Q, R)
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    return K, P


def finite_horizon_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                       N: int, Qf: np.ndarray | None = None) -> List[np.ndarray]:
    """
    Finite-horizon, time-varying LQR via backward Riccati recursion.

    Args:
        A, B: system matrices
        Q, R: stage cost
        N: horizon length
        Qf: terminal cost (defaults to Q)

    Returns:
        K_seq: list of gains [K_0, ..., K_{N-1}] for u_k = -K_k x_k
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    n = A.shape[0]
    P = np.asarray(Q if Qf is None else Qf, dtype=float)
    K_seq: list[np.ndarray] = [np.zeros((B.shape[1], n)) for _ in range(N)]

    for k in reversed(range(N)):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        K_seq[k] = K
        P = Q + A.T @ (P - P @ B @ K) @ A
    return K_seq
