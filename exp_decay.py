from scipy.integrate import solve_ivp


class ExponentialDecay:
    def __init__(self, a):
        self.a = a

    def __call__(self, t, u):
        """Return RHS of ODE."""
        return -self.a*u
