from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from math import sin

# Define the circuit class
class circuit():
    def __init__(self, R=10, L=20, C=0.05, A=20, w=20, p=0):
        self.R = R
        self.L = L
        self.C = C
        self.A = A
        self.w = w
        self.p = p
        self.t = None
        self.X = None

    # Define the ODE system for the circuit
    def ode_system(self, t, X):
        i1, i2, vc = X
        v_t = self.A * sin(self.w * t + self.p)
        common_term = (v_t - self.R * (i1 - i2)) / self.L
        di1_dt = common_term
        di2_dt = common_term - i2 / (self.R * self.C)
        dvc_dt = -i2 / self.C
        return [di1_dt, di2_dt, dvc_dt]

    # Simulate the circuit over a specified time period
    def simulate(self, t=10, pts=500):
        time = np.linspace(0, t, pts)
        X0 = [0, 0, 0]
        solution = solve_ivp(self.ode_system, (0, t), X0, t_eval=time, method='RK45')
        self.t = solution.t
        self.X = solution.y

    # Plot the results
    def doPlot(self):
        fig, ax = plt.subplots()

        i1 = self.X[0]
        i2 = self.X[1]
        vc = self.X[2]

        # Plot currents
        ax.plot(self.t, i1, 'k-', label=r'$i_1(t)$')
        ax.plot(self.t, i2, 'k--', label=r'$i_2(t)$')

        ax.set_xlim(min(self.t), max(self.t))
        ax.set_ylim(min(i1.min(), i2.min()) - 0.01, max(i1.max(), i2.max()) + 0.01)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(r"$i_1, i_2$ (A)")
        ax.grid(True, linestyle='--', linewidth=0.5)

        # Plot capacitor voltage on a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(self.t, vc, 'k:', label=r'$v_C(t)$')
        ax2.set_ylim(vc.min() - 0.01, vc.max() + 0.01)
        ax2.set_ylabel(r"$v_C(t)$ (V)")

        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.show()

# Create circuit instance with default parameters and simulate
CircuitObj = circuit()
CircuitObj.simulate(t=10, pts=500)
CircuitObj.doPlot()
