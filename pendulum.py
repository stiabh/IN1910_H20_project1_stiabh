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
        """Return array with time points on (0, T], in steps of dt."""
        if hasattr(self, "_t"):
            return self._t
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def theta(self):
        """Return array with theta as a function of t."""
        if hasattr(self, "_y"):
            return self._y[0]
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def omega(self):
        """Return array with omega as a function of t."""
        if hasattr(self, "_y"):
            return self._y[1]
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def x(self):
        """Return array with pendulum x-coordinates as a function of t."""
        return self.L*sin(self.theta)

    @property
    def y(self):
        """Return array with pendulum y-coordinates as a function of t."""
        return -self.L*cos(self.theta)

    @property
    def potential(self):
        """Return array with potential energy of pendulum as a function of t."""
        return self.M*self.g*(self.y + self.L)

    @property
    def vx(self):
        """Return array with pendulum velocity (x axis) as a function of t"""
        return np.gradient(self.x, self.t)

    @property
    def vy(self):
        """Return array with pendulum velocity (y axis) as a function of t"""
        return np.gradient(self.y, self.t)

    @property
    def kinetic(self):
        """Return array with kinetic energy of pendulum as a function of t"""
        return 0.5*self.M*(self.vx**2 + self.vy**2)

if __name__ == "__main__":
    pend = Pendulum(L=2.7)
    pend.solve((pi/6, 0.15), 10, 0.05)
    plt.plot(pend.t, pend.theta, label=r"$\theta(t)$")
    plt.title(r"Angle $\theta$ as a function of time")
    plt.xlabel(r"$t$"); plt.ylabel(r"$\theta$")
    plt.legend()

    plt.figure()
    plt.plot(pend.t, pend.potential, label=r"$P(t)$")
    plt.title(r"Potential energy $P$ as a function of time")
    plt.xlabel(r"$t$"); plt.ylabel("Energy (J)")
    plt.legend()

    plt.figure()
    plt.plot(pend.t, pend.kinetic, label=r"$K(t)$")
    plt.title(r"Kinetic energy $K$ as a function of time")
    plt.xlabel(r"$t$"); plt.ylabel("Energy (J)")
    plt.legend()

    plt.figure()
    plt.plot(pend.t, pend.potential+pend.kinetic,
             label=r"$E(t)$")
    plt.title(r"Total energy $E = P + K$ as a function of time")
    plt.xlabel(r"$t$"); plt.ylabel("Energy (J)")
    plt.legend()
    plt.show()

    """The last graph should be constant, i.e. a line, 
    but varies because of roundoff errors.
    """