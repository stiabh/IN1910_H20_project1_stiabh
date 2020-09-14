from math import cos, sin

class Pendulum:
    def __init__(self, L=1, M=1, g=9.81):
        """Set pendulum parameters.

        L: length of rod [m]
        M: mass at end of rod [kg]
        g: acceleration of gravity [m/s**2]
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