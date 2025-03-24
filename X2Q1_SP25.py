# Updated full code with fixed x-axis ticks and cleanup

import numpy as np
from scipy.integrate import quad, solve_ivp
from matplotlib import pyplot as plt

# Define the Fresnel integral function
def S(x):
    result = quad(lambda t: np.sin(t**2), 0, x)
    return result[0]  # Return only the value of the integral

# Define the exact solution
def Exact(x):
    return 1 / (2.5 - S(x)) + 0.01 * x**2

# Define the ODE system
def ODE_System(x, y):
    Y = y[0]
    Ydot = (Y - 0.01 * x**2)**2 * np.sin(x**2) + 0.02 * x
    return [Ydot]

# Function to plot the results
def Plot_Result(xRange_Num, y_Num, xRange_Xct, y_Xct):
    plt.figure(figsize=(8, 5))
    # Numerical solution (upward facing triangles)
    plt.plot(xRange_Num, y_Num, '^', label='Numerical', markersize=5, color='black')
    # Exact solution (solid line)
    plt.plot(xRange_Xct, y_Xct, label='Exact', linestyle='-', color='black')

    # Formatting
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 6)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0.0, 7.0, 1.0))  # Set integer ticks from 0 to 6
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title(r"IVP: $y' = (y - 0.01x^2)^2 \sin(x^2) + 0.02x, y(0) = 0.4$")
    plt.legend()
    plt.grid(True)

    plt.show()

# Main function to solve and plot
def main():
    xRange = np.arange(0, 6.2, 0.2)  # Step size of 0.2
    xRange_xct = np.linspace(0, 6, 500)  # For smooth exact solution
    Y0 = [0.4]  # Initial condition y(0) = 0.4

    # Solve the initial value problem using solve_ivp
    sln = solve_ivp(ODE_System, [0, 6], Y0, t_eval=xRange)

    # Compute the exact solution over the exact x range
    xctSln = np.array([Exact(x) for x in xRange_xct])

    # Plot results
    Plot_Result(sln.t, sln.y[0], xRange_xct, xctSln)

# Run the main function
main()
