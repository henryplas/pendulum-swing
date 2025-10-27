import numpy as np
from msdcontrol.env import MassSpringDamperEnv, MassSpringDamperParams
from msdcontrol.lqr import dlqr

def test_dlqr_stabilizes_msd():
    env = MassSpringDamperEnv(MassSpringDamperParams(m=1.0, k=2.0, c=0.4), dt=0.02)
    A, B = env.Ad, env.Bd
    assert A is not None and B is not None

    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])

    K, P = dlqr(A, B, Q, R)  # noqa: F841

    Acl = A - B @ K
    eigs = np.linalg.eigvals(Acl)
    assert np.all(np.abs(eigs) < 1.0), "Closed-loop eigenvalues must lie inside unit circle for DT stability."
