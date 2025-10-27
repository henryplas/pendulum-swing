"""
Mass–Spring–Damper environment and utilities.

- Continuous-time model: x_dot = A x + B u
- Discretization: Zero-Order Hold (preferred, requires SciPy); fallback to Euler
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    from scipy.linalg import expm  # type: ignore
    _HAVE_EXPM = True
except Exception:  # noqa: BLE001
    expm = None
    _HAVE_EXPM = False


def zoh_discretize(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-Order Hold discretization.
    If scipy.linalg.expm is unavailable, falls back to forward Euler.

    Args:
        A: (n, n) continuous-time state matrix
        B: (n, m) continuous-time input matrix
        dt: step size

    Returns:
        (Ad, Bd): discrete-time matrices
    """
    n = A.shape[0]
    if _HAVE_EXPM:
        M = np.zeros((n + B.shape[1], n + B.shape[1]))
        M[:n, :n] = A
        M[:n, n:] = B
        Md = expm(M * dt)
        Ad = Md[:n, :n]
        Bd = Md[:n, n:]
        return Ad, Bd
    else:
        # Forward Euler as a simple fallback (accurate for small dt)
        Ad = np.eye(n) + A * dt
        Bd = B * dt
        return Ad, Bd


@dataclass
class MassSpringDamperParams:
    m: float = 1.0  # mass
    k: float = 1.0  # spring constant
    c: float = 0.2  # damping coefficient


class MassSpringDamperEnv:
    """
    Minimal mass–spring–damper environment focused on control experimentation.

    Continuous-time:
        x = [position, velocity]
        x_dot = [v, (1/m)(-k * p - c * v + u)]

    Discrete-time:
        x_{k+1} = Ad x_k + Bd u_k
    """

    def __init__(
        self,
        params: MassSpringDamperParams = MassSpringDamperParams(),
        dt: float = 0.01,
        process_noise_std: float | np.ndarray | None = None,
        u_limit: Optional[float] = None,
        discretize: bool = True,
    ) -> None:
        self.params = params
        self.dt = float(dt)
        self.process_noise_std = process_noise_std
        self.u_limit = u_limit

        m, k, c = params.m, params.k, params.c

        # Continuous-time matrices
        self.A = np.array([[0.0, 1.0],
                           [-k / m, -c / m]], dtype=float)
        self.B = np.array([[0.0],
                           [1.0 / m]], dtype=float)

        # Discrete-time matrices (ZOH by default)
        if discretize:
            self.Ad, self.Bd = zoh_discretize(self.A, self.B, self.dt)
        else:
            self.Ad, self.Bd = None, None  # type: ignore

        self.x = np.zeros(2, dtype=float)

    # -------------------- API -------------------- #
    def reset(self, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset the state. Returns the state."""
        if x0 is None:
            self.x = np.zeros(2, dtype=float)
        else:
            self.x = np.asarray(x0, dtype=float).reshape(2)
        return self.x.copy()

    def set_state(self, x: np.ndarray) -> None:
        """Manually set the internal state."""
        self.x = np.asarray(x, dtype=float).reshape(2)

    def step(self, u: float) -> np.ndarray:
        """
        Advance one discrete step using DT matrices (Ad, Bd).

        Args:
            u: control input (scalar). Clamped to u_limit if provided.

        Returns:
            next state
        """
        if self.Ad is None or self.Bd is None:
            raise RuntimeError("This environment was created with discretize=False. "
                               "Use .step_dt() or recreate with discretize=True.")

        u = float(u)
        if self.u_limit is not None:
            u = np.clip(u, -self.u_limit, self.u_limit)

        x_next = self.Ad @ self.x + self.Bd.flatten() * u

        # Optional process noise
        if self.process_noise_std is not None:
            std = np.asarray(self.process_noise_std, dtype=float).reshape(-1)
            if std.size == 1:
                noise = np.random.randn(*self.x.shape) * std.item()
            else:
                noise = np.random.randn(*self.x.shape) * std
            x_next = x_next + noise

        self.x = x_next
        return self.x.copy()

    def xdot(self, x: np.ndarray, u: float) -> np.ndarray:
        """Continuous-time dynamics x_dot = A x + B u."""
        x = np.asarray(x, dtype=float).reshape(2)
        return self.A @ x + self.B.flatten() * float(u)

    def step_ct(self, u: float) -> np.ndarray:
        """
        Advance one step using forward-Euler on CT dynamics.
        Useful when Ad, Bd are not precomputed.
        """
        u = float(u)
        if self.u_limit is not None:
            u = np.clip(u, -self.u_limit, self.u_limit)
        self.x = self.x + self.dt * self.xdot(self.x, u)
        return self.x.copy()
