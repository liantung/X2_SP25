import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

# Function to calculate S(x) using quad
def S(x):
    s = quad(lambda t: np.sin(t ** 2), 0, x) # the solution for S(x) found using quad
    return s[0]

# Function for the exact solution
def Exact(x):
    return (1 / (2.5 - S(x))) + 0.01 * x ** 2

# Define the ODE system
def ODE_System(x, y):
    Ydot = ((y[0] - 0.01 * x ** 2) ** 2) * np.sin(x ** 2) + 0.02 * x
    return [Ydot]

# Plot function
def Plot_Result(xRange_Num, y_Num, xRange_Xct, y_Xct):
    plt.figure(figsize=(8, 5))
    plt.plot(xRange_Xct, y_Xct, 'k-', label='Exact')  # Solid line for exact solution
    plt.plot(xRange_Num, y_Num, 'r^', markerfacecolor='none',
             label='Numerical')  # Triangles for numerical solution

    plt.xlim(0.0, 6.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(np.arange(0, 6.2, 1),
               labels=[f"{val:.1f}" for val in np.arange(0, 6.2, 1)])
    plt.yticks(np.arange(0, 1.2, 0.2), labels=[f"{val:.1f}" for val in np.arange(0, 1.2, 0.2)])
    plt.legend()
    plt.title(r"IVP: $y' = (y - 0.01x^2)^2 \sin(x^2) + 0.02x$, $y(0) = 0.4$", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Main function to solve and plot the ODE
def main():
    xRange = np.arange(0, 6.2, 0.2) # x range for numerical solution
    xRange_xct = np.linspace(0, 6, 600) # x range for exact solution
    Y0 = [0.4]  # Initial condition

    # Solve the ODE using RK45
    sln = solve_ivp(ODE_System, [0, 6], Y0, t_eval=xRange, method='RK45', max_step=0.2)

    # Generate exact solution
    xctSln = np.array([Exact(x) for x in xRange_xct])

    # Plot the result
    Plot_Result(xRange, sln.y[0], xRange_xct, xctSln)

# Run the main function
main()
