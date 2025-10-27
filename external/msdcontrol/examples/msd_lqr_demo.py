"""
Run a simple LQR on the mass–spring–damper and plot results.
"""

import numpy as np
import matplotlib.pyplot as plt

from msdcontrol.env import MassSpringDamperEnv, MassSpringDamperParams
from msdcontrol.lqr import dlqr
from msdcontrol.sim import simulate, lqr_controller


def main() -> None:
    # Environment and discretization
    params = MassSpringDamperParams(m=1.0, k=2.0, c=0.4)
    env = MassSpringDamperEnv(params=params, dt=0.01, u_limit=5.0, discretize=True)

    A, B = env.Ad, env.Bd  # use discrete-time matrices
    assert A is not None and B is not None

    # LQR weights (tune these)
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])

    # Optimal infinite-horizon dlqr
    K, P = dlqr(A, B, Q, R)  # noqa: F841

    # Simulate
    x0 = np.array([1.0, 0.0])  # 1 meter initial displacement
    steps = 1000
    xs, us, cost = simulate(A, B, lqr_controller(K), x0, steps, Q=Q, R=R)

    print(f"Total quadratic cost over {steps} steps: {cost:.3f}")

    # Time vector
    t = np.arange(steps + 1) * env.dt

    # Plots
    plt.figure()
    plt.plot(t, xs[:, 0], label="position p")
    plt.plot(t, xs[:, 1], label="velocity v")
    plt.xlabel("time [s]")
    plt.ylabel("state")
    plt.legend()
    plt.title("MSD states under LQR")

    plt.figure()
    plt.step(t[:-1], us, where="post")
    plt.xlabel("time [s]")
    plt.ylabel("control u")
    plt.title("LQR control input")

    plt.show()


if __name__ == "__main__":
    main()
