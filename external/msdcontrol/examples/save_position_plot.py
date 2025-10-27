
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from msdcontrol.env import MassSpringDamperEnv, MassSpringDamperParams
from msdcontrol.lqr import dlqr
from msdcontrol.sim import simulate, lqr_controller

def main() -> None:
    params = MassSpringDamperParams(m=1.0, k=2.0, c=0.4)
    env = MassSpringDamperEnv(params=params, dt=0.01, u_limit=5.0, discretize=True)

    A, B = env.Ad, env.Bd
    assert A is not None and B is not None

    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])

    K, _ = dlqr(A, B, Q, R)
    x0 = np.array([1.0, 0.0])
    steps = 1000

    xs, _, _ = simulate(A, B, lqr_controller(K), x0, steps, Q=Q, R=R)
    t = np.arange(steps + 1) * env.dt

    plt.figure()
    plt.plot(t, xs[:, 0])
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Position vs Time (LQR)")
    (PROJECT_ROOT / "outputs").mkdir(exist_ok=True)
    out_path = PROJECT_ROOT / "outputs" / "position_vs_time.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
