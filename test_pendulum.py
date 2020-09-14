from pendulum import *
from numpy import pi

def test_Pendulum_params():
    tol = 1e-10
    pend = Pendulum(L=2.7)
    theta = pi/6
    omega = 0.15
    dtheta_dt, domega_dt = pend(0, (theta, omega))
    assert abs(dtheta_dt - 0.15) < tol
    assert abs(domega_dt + 109/60) < tol

def test_Pendulum_params_at_rest():
    pend = Pendulum()
    theta = 0
    omega = 0
    dtheta_dt, domega_dt = pend(0, (theta, omega))
    assert dtheta_dt == 0
    assert domega_dt == 0