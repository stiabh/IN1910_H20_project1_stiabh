from exp_decay import *

def test_ExponentialDecay():
    tol = 1e-10
    ode = ExponentialDecay(0.4)
    assert abs(ode(0, 3.2) + 1.28) < tol