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
                         theta2_0, omega2_0), t_eval=_t_eval)
        self._t, self._y = _sol.t, _sol.y
