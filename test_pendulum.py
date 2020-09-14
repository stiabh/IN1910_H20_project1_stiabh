import pytest
import numpy as np
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


def test_Pendulum_property_exceptions():
    pend = Pendulum()
    with pytest.raises(AttributeError):
        pend.t
        pend.theta
        pend.omega

def test_Pendulum_theta_omega_array_values():
    pend = Pendulum()
    y0 = (0, 0)
    T = 10
    dt = 0.1
    pend.solve(y0, T, dt)
    assert np.all(pend.t == np.arange(0, T+dt, dt))
    assert np.all(pend.theta == 0)
    assert np.all(pend.omega == 0)