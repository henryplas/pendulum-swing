# run_pendulum.py
# Trains a coarse value function for swing-up (DP on a grid),
# then hands off to LQR near upright to balance.

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt

from pendulum_env import PendulumEnv, PendulumParams
from dp_tool import value_iteration, greedy_from_grid, wrap_pi
from msdcontrol.lqr import dlqr  

def stage_cost(x, u, q_theta=4.0, q_dtheta=0.5, r=0.05):
    theta, dtheta = x
    phi = wrap_pi(theta - math.pi)
    return q_theta*phi*phi + q_dtheta*dtheta*dtheta + r*u*u

def main():
    env = PendulumEnv(PendulumParams(u_limit=3.0))
    dt = 0.02

    TH_POINTS   = 49
    DTH_POINTS  = 37
    ACT_POINTS  = 11

    th_grid  = np.linspace(-math.pi, math.pi, TH_POINTS)
    dth_grid = np.linspace(-8.0, 8.0, DTH_POINTS)
    actions  = np.linspace(-env.p.u_limit, env.p.u_limit, ACT_POINTS)

    print("Running value iteration")
    V, Pi = value_iteration(
        step_fn=env.rk4,
        stage_cost=lambda x,u: stage_cost(x,u),
        th_grid=th_grid,
        dth_grid=dth_grid,
        actions=actions,
        dt=dt,
        gamma=0.995,
        max_iters=300,
        tol=2e-4,
    )
    print("DP done.")

    A, B = env.linearize_upright_ct()
    Ad, Bd = env.discretize(A, B, dt)


    # Penalties for (phi, dphi) about upright
    Q = np.diag([20.0, 3.0])
    R = np.array([[0.2]])
    K, S = dlqr(Ad, Bd, Q, R)  # expects shape (1,2) for K

    T = 8.0
    steps = int(T/dt)
    x = np.array([0.0, 0.0])   # start hanging down
    xs = [x.copy()]
    us = []

    def near_upright(x):
        phi = wrap_pi(x[0] - math.pi)
        return (abs(phi) < 0.2) and (abs(x[1]) < 1.0)

    for k in range(steps):
        if near_upright(x):
            # switch to your LQR around upright; state = [phi, dphi]
            phi = wrap_pi(x[0] - math.pi)
            x_lqr = np.array([phi, x[1]])
            u = -(K @ x_lqr).item()

        else:
            # greedy action from DP policy grid
            u = greedy_from_grid(x, Pi, th_grid, dth_grid)

        # clamp to env limits
        u = float(np.clip(u, -env.p.u_limit, env.p.u_limit))
        us.append(u)

        x = env.rk4(x, u, dt)
        xs.append(x.copy())

    xs = np.array(xs)
    us = np.array(us)

    # Save an animation of the swing-up (theta over time) to outputs/pendulum_swing.gif
    gif_path = env.save_gif(
        thetas=xs[:, 0],           # use theta sequence from the sim
        path="outputs/pendulum_swing.gif",
        dt=dt,
        fps=None,                  # None -> uses round(1/dt)
        dpi=140,
        trail=True                 # set False if you prefer no trail
    )
    print(f"Saved pendulum animation to: {gif_path}")


    # t = np.arange(xs.shape[0]) * dt
    # fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    # ax[0].plot(t, np.unwrap(xs[:,0])) ; ax[0].set_ylabel("theta [rad]")
    # ax[1].plot(t, xs[:,1])            ; ax[1].set_ylabel("dtheta [rad/s]")
    # ax[2].plot(t[:-1], us)            ; ax[2].set_ylabel("u [N·m]") ; ax[2].set_xlabel("time [s]")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
