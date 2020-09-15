import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi
from scipy.integrate import solve_ivp


class DoublePendulum:
    def __init__(self, M1=1, L1=1, M2=1, L2=1, g=9.81):
        """Set parameters of pendulums.
        
        First pendulum (attached to stationary point):
        L1: Length of rod [m]
        M1: Mass at end of rod [kg]

        Second pendulum (attached to first pendulum):
        L2: Length of rod [m]
        M2: Mass at end of rod [kg]

        g: Acceleration of gravity [m/s**2]
        """
        self.M1 = M1
        self.L1 = L1
        self.M2 = M2
        self.L2 = L2
        self.g = g

    def __call__(self, t, y):
        """Return RHS of ODEs."""
        theta1, omega1, theta2, omega2 = y[0], y[1], y[2], y[3]
        M1, L1, M2, L2 = self.M1, self.L1, self.M2, self.L2
        g = self.g
        delta_t = theta2 - theta1

        dtheta1_dt = omega1
        domega1_dt = (M2*L1*omega1**2*sin(delta_t)*cos(delta_t)
                   + M2*g*sin(theta2)*cos(delta_t)
                   + M2*L2*omega2**2*sin(delta_t)
                   - (M1+M2)*g*sin(theta1)) / ((M1+M2)*L1
                                               - M2*L1*cos(delta_t)**2)

        dtheta2_dt = omega2
        domega2_dt = (-M2*L2*omega2**2*sin(delta_t)*cos(delta_t)
                   + (M1+M2)*g*sin(theta1)*cos(delta_t)
                   - (M1+M2)*L1*omega1**2*sin(delta_t)
                   - (M1+M2)*g*sin(theta2)) / ((M1+M2)*L2
                                               - M2*L2*cos(delta_t)**2)

        return dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt

    def solve(self, y0, T, dt, angles="rad"):
        """Solve set of ODEs using scipy.

        y0: Tuple (theta1_0, omega1_0, theta2_0, omega2_0) with 
            initial conditions
        T: End of interval (0, T)
        dt: Time step for interval
        angles: deg (degrees) or rad (radians)
        """
        theta1_0, omega1_0, theta2_0, omega2_0 = y0[0], y0[1], y0[2], y0[3]

        if angles == "rad":
            pass
        elif angles == "deg":
            theta1_0 = theta1_0*pi/180
            omega1_0 = omega1_0*pi/180
            theta2_0 = theta2_0*pi/180
            omega2_0 = omega2_0*pi/180
        else:
            raise ValueError("Angles must be set to deg or rad")

        _t_eval = np.arange(0, T+dt, dt)
        _sol = solve_ivp(self.__call__, (0, T), (theta1_0, omega1_0, 
                         theta2_0, omega2_0), t_eval=_t_eval, method="Radau")
        self._t, self._y = _sol.t, _sol.y

    @property
    def t(self):
        """Return array with time points on (0, T], in steps of dt."""
        if hasattr(self, "_t"):
            return self._t
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def theta1(self):
        """Return array with theta as a function of t."""
        if hasattr(self, "_y"):
            return self._y[0]
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def theta2(self):
        """Return array with theta as a function of t."""
        if hasattr(self, "_y"):
            return self._y[2]
        else:
            raise AttributeError("Method solve has not been called")

    @property
    def x1(self):
        """Return array with 1st pendulum x-coord. as a function of t."""
        return self.L1*sin(self.theta1)

    @property
    def y1(self):
        """Return array with 1st pendulum y-coord. as a function of t."""
        return -self.L1*cos(self.theta1)

    @property
    def x2(self):
        """Return array with 2nd pendulum x-coord. as a function of t."""
        return self.x1 + self.L2*sin(self.theta2)

    @property
    def y2(self):
        """Return array with 2nd pendulum y-coord. as a function of t."""
        return self.y1 - self.L2*cos(self.theta2)

    @property
    def potential(self):
        """Return array with total potential energy as a function of t."""
        P1 = self.M1*self.g*(self.y1+self.L1)
        P2 = self.M2*self.g*(self.y2+self.L1+self.L2)
        return P1 + P2

    @property
    def vx1(self):
        """Return array with 1st pendulum velocity (x axis).|"""
        return np.gradient(self.x1, self.t)

    @property
    def vy1(self):
        """Return array with 1st pendulum velocity (y axis)."""
        return np.gradient(self.y1, self.t)

    @property
    def vx2(self):
        """Return array with 2nd pendulum velocity (x axis)."""
        return np.gradient(self.x2, self.t)

    @property
    def vy2(self):
        """Return array with 2nd pendulum velocity (y axis)."""
        return np.gradient(self.y2, self.t)

    @property
    def kinetic(self):
        """Return array with total kinetic energy."""
        K1 = 0.5*self.M1*(self.vx1**2 + self.vy1**2)
        K2 = 0.5*self.M2*(self.vx2**2 + self.vy2**2)
        return K1 + K2

if __name__ == "__main__":
    pend = DoublePendulum()
    pend.solve((0, 0.15, 0, 0.15), 10, 0.01)

    plt.plot(pend.t, pend.potential, label=r"$P(t)$")
    plt.title(r"Total potential energy $P$ as a function of time")
    plt.xlabel(r"$t$")
    plt.ylabel("Energy (J)")
    plt.legend()

    plt.figure()
    plt.plot(pend.t, pend.kinetic, label=r"$K(t)$")
    plt.title(r"Total kinetic energy $K$ as a function of time")
    plt.xlabel(r"$t$")
    plt.ylabel("Energy (J)")
    plt.legend()

    plt.figure()
    plt.plot(pend.t, pend.potential+pend.kinetic, label=r"$E(t)$")
    plt.title(r"Total energy $E = P + K$ as a function of time")
    plt.xlabel(r"$t$")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.show()