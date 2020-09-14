import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi
from scipy.integrate import solve_ivp


class Pendulum:
    def __init__(self, L=1, M=1, g=9.81):
        """Set pendulum parameters.

        L: Length of rod [m]
        M: Mass at end of rod [kg]
        g: Acceleration of gravity [m/s**2]
        """

        self.L = L
        self.M = M
        self.g = g

    def __call__(self, t, y):
        """Return RHS of ODEs."""
        _theta = y[0]
        _omega = y[1]

        _dtheta_dt = _omega
        _domega_dt = -self.g/self.L*sin(_theta)

        return _dtheta_dt, _domega_dt

    def solve(self, y0, T, dt, angles="rad"):
        """Solve set of ODEs using scipy.

        y0: Tuple (theta0, omega0) with initial conditions
        T: End of interval (0, T)
        dt: Time step for interval
        angles: deg (degrees) or rad (radians)
        """
        _theta0, _omega0 = y0[0], y0[1]

        if angles == "rad":
            pass
        elif angles == "deg":
            _theta0 = _theta0*pi/180
            _omega0 = _omega0*pi/180
        else:
            raise ValueError("Angles must be set to deg or rad")

        _t_eval = np.arange(0, T+dt, dt)
        _sol = solve_ivp(self.__call__, (0, T), (_theta0, _omega0),
                         t_eval=_t_eval)
        self._t, self._y = _sol.t, _sol.y

    @property
    def t(self):
        if hasattr(self, "_t"):
            return self._t
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def theta(self):
        if hasattr(self, "_y"):
            return self._y[0]
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def omega(self):
        if hasattr(self, "_y"):
            return self._y[1]
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def x(self):
        return self.L*sin(self.theta)

    @property
    def y(self):
        return -self.L*cos(self.theta)