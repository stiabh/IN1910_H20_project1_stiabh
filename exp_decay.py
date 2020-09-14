import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class ExponentialDecay:
    def __init__(self, a):
        self.a = a

    def __call__(self, t, u):
        """Return RHS of ODE."""
        return -self.a*u

    def solve(self, u0, T, dt):
        """Solve ODE for initial condition u0 on (0, T], in steps of dt.
        Returns two arrays, one for time points and one for solution points.
        """
        _t_eval = np.arange(0, T+dt, dt)
        _sol = solve_ivp(self.__call__, (0, T), (u0, ), t_eval=_t_eval)
        return _sol.t, _sol.y[0]


if __name__ == "__main__":
    a = 0.4
    u0 = 3.2
    T = 20
    dt = 0.1

    decay_model = ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)

    plt.plot(t, u, label=r"$a = 0.4, u_0 = 3.2$")
    plt.legend()
    plt.show()
