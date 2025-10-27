# pendulum_env.py
# Simple torque-actuated pendulum with viscous friction.
# Provides: continuous dynamics, RK4 step, linearization about upright,
# and exact ZOH discretization for LQR use.

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
from matplotlib import animation


try:
    from scipy.linalg import expm
    _HAS_EXPM = True
except Exception:
    _HAS_EXPM = False

@dataclass
class PendulumParams:
    m: float = 1.0      # mass (kg)
    L: float = 1.0      # length (m)
    b: float = 0.05     # viscous friction (N*m*s/rad)
    g: float = 9.81     # gravity (m/s^2)
    u_limit: float = 3.0

class PendulumEnv:
    """
    State x = [theta, dtheta], torque input u (N*m).
    Upright is theta = pi (we'll regulate phi = theta - pi).
    """
    def __init__(self, params: PendulumParams = PendulumParams()):
        self.p = params
        self.I = self.p.m * self.p.L * self.p.L

    @staticmethod
    def wrap_pi(angle: float | np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(angle), np.cos(angle))

    def f(self, x: np.ndarray, u: float) -> np.ndarray:
        """Continuous dynamics xdot = [dtheta, ddtheta]."""
        theta, dtheta = x
        m, L, b, g, I = self.p.m, self.p.L, self.p.b, self.p.g, self.I
        ddtheta = (u - b * dtheta - m * g * L * np.sin(theta)) / I
        return np.array([dtheta, ddtheta])

    def rk4(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        """One RK4 step."""
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def linearize_upright_ct(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Linearize about upright equilibrium (theta = pi, dtheta = 0).
        Use coordinates phi = theta - pi (so equilibrium at 0).
        x = [phi, dphi], u = torque.
        """
        m, L, b, g, I = self.p.m, self.p.L, self.p.b, self.p.g, self.I
        A = np.array([[0.0,      1.0],
                      [ m*g*L/I, -b/I]])
        B = np.array([[0.0],
                      [1.0 / I]])
        return A, B

    def discretize(self, A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Exact ZOH using matrix exponential if SciPy is present; else Euler.
        """
        n = A.shape[0]
        if _HAS_EXPM:
            M = np.zeros((n + B.shape[1], n + B.shape[1]))
            M[:n, :n] = A
            M[:n, n:] = B
            Md = expm(M * dt)
            Ad = Md[:n, :n]
            Bd = Md[:n, n:]
            return Ad, Bd
        else:
            # Forward Euler fallback (good for small dt)
            Ad = np.eye(n) + dt * A
            Bd = dt * B
            return Ad, Bd

    def save_gif(self, thetas, path="outputs/pendulum_swing.gif", dt=0.02, fps=None, dpi=120, trail=False):
        """
        Save an animated GIF of the pendulum motion.
        Parameters
        thetas : array-like
            Sequence of theta angles [rad] for each frame.
        path : str
            Output GIF path (folders will be created if missing).
        dt : float
            Simulation time step [s].
        fps : int or None
            Frames per second for GIF. If None, uses round(1/dt).
        dpi : int
            DPI of the rendered GIF.
        trail : bool
            If True, draws a short trail behind the bob.
        """
        L = self.p.L
        thetas = np.asarray(thetas, dtype=float)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # derive fps if not given
        if fps is None:
            fps = max(1, int(round(1.0 / dt)))

        # world coords (pivot at origin)
        def bob_xy(theta):
            # x right, y up; bob hangs down for theta=0 (y = -L)
            x = L * np.sin(theta)
            y = -L * np.cos(theta)
            return x, y

        # figure
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        margin = 1.2 * L
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Pendulum swing")

        # graphics: pivot, rod, bob, optional trail
        pivot, = ax.plot([0], [0], marker='o', ms=6, color='black')
        rod,   = ax.plot([], [], lw=3)
        bob,   = ax.plot([], [], marker='o', ms=12)
        if trail:
            trail_len = max(3, int(0.4 / dt))  # ~0.4s
            trail_line, = ax.plot([], [], lw=1.5, alpha=0.7)
        else:
            trail_line = None

        # init function for FuncAnimation
        def init():
            rod.set_data([], [])
            bob.set_data([], [])
            if trail_line is not None:
                trail_line.set_data([], [])
            return (rod, bob) if trail_line is None else (rod, bob, trail_line)

        # per-frame update
        xs, ys = [], []
        def update(i):
            theta = thetas[i]
            x, y = bob_xy(theta)
            rod.set_data([0.0, x], [0.0, y])
            bob.set_data([x], [y])

            if trail_line is not None:
                xs.append(x); ys.append(y)
                xs_trim = xs[-trail_len:]; ys_trim = ys[-trail_len:]
                trail_line.set_data(xs_trim, ys_trim)

            return (rod, bob) if trail_line is None else (rod, bob, trail_line)

        # animate & save
        anim = animation.FuncAnimation(fig, update, init_func=init,
                                       frames=len(thetas),
                                       interval=1000.0*dt, blit=True)
        writer = animation.PillowWriter(fps=fps)
        anim.save(path, writer=writer, dpi=dpi)
        plt.close(fig)
        return path
