import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

#rewriting the system so Python can understand it
def dSdt(t, S):
    #C and beta DO NOT CHANGE. System is stable for every C in [0,sqrt(2)]
    C = 1.35
    beta = 4

    #change H and gamma for arrhythmias and HR change, respectively
    H = 3
    gamma = 5
    x1, x2, x3, x4 = S
    return [gamma*(x1 - x2 - C*x1*x2 - x1*(x2**2)),
            gamma*(H*x1 - 3*x2 + C*x1*x2 + x1*(x2**2) + beta*(x4-x2)),
            gamma*(x3 - x4 - C*x3*x4 - x3*(x4**2)),
            gamma*(H*x3 - 3*x4 + C*x3*x4 + x3*(x4**2) + 2*beta*(x2-x4))]

#initial conditions
#stationary point is at 0,0,0,0 so we have to give it an initial "kick"
x1_0 = 0
x2_0 = 0
x3_0 = 0.1
x4_0 = 0

#initial condition vector
S_0 = (x1_0, x2_0, x3_0, x4_0)

#solving the ODE's
t = np.linspace(0, 6, 100000)
sol = odeint(dSdt, y0=S_0, t=t, tfirst=True)

#extracting the results
x1_sol = sol.T[0]
x2_sol = sol.T[1]
x3_sol = sol.T[2]
x4_sol = sol.T[3]

#coefficients for linear combination
alpha_1 = -0.024
alpha_2 = 0.0216
alpha_3 = -0.0012
alpha_4 = 0.12

#the final solution to our problem
EKG = alpha_1*x1_sol + alpha_2*x2_sol + alpha_3*x3_sol + alpha_4*x4_sol

#wow pretty pictures :-)

plt.plot(t, x1_sol)
plt.plot(t, x2_sol)
#plt.plot(t, x3_sol)
#plt.plot(t, x4_sol)
#plt.plot(t, EKG)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [arbitrary units]")
plt.title("A synthesised EKG signal")
plt.show()
#plt.savefig("myECGplot.png")