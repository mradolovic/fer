import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dSdt(t, S):
    global H
    global gamma

    #C and beta DO NOT CHANGE. System is stable for every C in [0,sqrt(2)]
    c = 1.35
    beta = 4

    #change H and gamma for arrhythmias and HR change, respectively
    #H = 3
    #gamma = 5
    x1, x2, x3, x4 = S
    return [gamma*(x1 - x2 - c*x1*x2 - x1*(x2**2)),
            gamma*(H*x1 - 3*x2 + c*x1*x2 + x1*(x2**2) + beta*(x4-x2)),
            gamma*(x3 - x4 - c*x3*x4 - x3*(x4**2)),
            gamma*(H*x3 - 3*x4 + c*x3*x4 + x3*(x4**2) + 2*beta*(x2-x4))]

# Initial conditions: x1(0) = 1, x2(0) = 0, ...
initial_conditions = [0, 0, 0.1, 0]

#Change H and gamma for sinus rhythm or arrhythmias
H = 3
gamma = 5

# Time span for the solution
t_span = (0, 10)  # From t=0 to t=10
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Points at which to evaluate the solution

# Solve the system using solve_ivp
solution = solve_ivp(
    dSdt,
    t_span,
    initial_conditions,
    t_eval=t_eval,
    method='RK45'
)

# Extract the solution
t = solution.t
x1_sol = solution.y[0]
x2_sol = solution.y[1]
x3_sol = solution.y[2]
x4_sol = solution.y[3]

#coefficients for linear combination
alpha_1 = -0.024
alpha_2 = 0.0216
alpha_3 = -0.0012
alpha_4 = 0.12

#the final solution to our problem
EKG = alpha_1*x1_sol + alpha_2*x2_sol + alpha_3*x3_sol + alpha_4*x4_sol

#wow pretty pictures :-)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t, EKG, label=r'$y_1(t)$ (Position)')
plt.title('Solution of the Harmonic Oscillator')
plt.xlabel('Time (t)')
plt.ylabel('Solution')
plt.legend()
plt.grid()
plt.show()
