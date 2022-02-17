import numpy as np
from matplotlib import pyplot as plt

#   Physical Constants:
h_bar = 1.05457e-34
MeV   = 1.60217648e-13
omega = 5.34e21                 # Harmonic oscillator frequency
energy_scale = 3.5

#   Calculation precision:
eps = 1.0e-11                   # Required precision for the solution to achieve
n_divisions = 1200              # Number of divisions of the integration region
n_energyLev = 5                 # Number of energy levels to compute
n_zeros = 0                     # Number of zeroes in the solution
x_range = 11.0                  # Integration region length
x_delta = x_range / n_divisions # Width of each division (spatial resolution)

#   Potential:
def E_pot_fun(x):
    return 0.5 * pow(x,2.0)     # Harmonic potential

region = np.arange(-x_range/2,x_range/2,x_range/n_divisions)
E_pot  = (np.vectorize(E_pot_fun))(region)

#   Boundary conditions:
psi_left  = 1.0e-3
psi_right = 0.0
end_sign  = -1
psi = np.zeros(n_divisions)
psi[0] = psi_left
psi[1] = psi_left + 1.0e-3
psi[-1] = psi_right

#   Solution searching params:
E_lowerBound = 0.0
E_upperBound = 10.0
E_trial = E_upperBound
E_EigenEnergies = np.zeros(n_energyLev)


#############
#   Solver  #
#############

def k_squared(i, E):
    return 2.0 * (E - E_pot[i])         # k^2(x) = 2m(E - V(x)) / hbar^2

def numerov_approx(psi_m1, psi_m2, i ,E):
    denom = 1.0 + (pow(x_delta,2) * k_squared(i,E) / 12.0)
    term1 = 1.0 - (5.0 * pow(x_delta,2) * k_squared(i-1,E) / 12.0)
    term2 = 1.0 + (pow(x_delta,2) * k_squared(i-2,E) / 12.0)
    return ((2.0 * term1 * psi_m1) - term2 * psi_m2) / denom

def coarse_search_bounds(ith_energyLev):
    """
    Iterative step in search of an energy upper bound.

    Parameters:
    ith_energyLev (int): the energy level of the solution

    Returns:
    bool: True if the bounds have the appropriate amount of zeros, false otherwise. 
    """
    global E_upperBound
    global E_lowerBound
    global n_zeros

    n_zeros = 0
    E_trial = E_upperBound

    #   Solve once:
    for i in range(2, n_divisions):
        psi[i] = numerov_approx(psi[i-1], psi[i-2], i, E_trial)
        #   Count the zeros of the solution:
        if (psi[i-1]*psi[i]) < 0.0 :
            n_zeros += 1

    if E_upperBound < E_lowerBound:
        E_upperBound = max(2.0 * E_upperBound, - 2.0 * E_upperBound)

    #   Find the right amount of nodes for the solution
    if n_zeros > ith_energyLev :
        E_upperBound *= 0.6
        return False
    elif n_zeros < ith_energyLev :
        E_upperBound *= 2.0
        return False
    else:
        return True

def refine_bounds(ith_energyLev):
    """
    Refines the energy level by looking at the right boundary condition.
    
    Parameters:
    ith_energyLev (int): the energy level of the searched solution.

    Returns: 
    bool: True if the precision goal is achieved, false otherwise.
    """
    global E_upperBound
    global E_lowerBound
    global n_zeros

    E_trial = (E_lowerBound + E_upperBound) /2

    #   Solve once:
    for i in range(2,n_divisions):
        psi[i] = numerov_approx(psi[i-1],psi[i-2],i,E_trial)

    if (end_sign * psi[-1]) > psi_right :
        E_lowerBound = E_trial
    else :
        E_upperBound = E_trial

    return (E_upperBound - E_lowerBound) < eps


################
#   Main Loop  #
################

for ith_energyLev in range(1,n_energyLev) :

    print("Energy level - {}: searching solution.".format(ith_energyLev))

    #   Fast search for an energy upper bound on the all eigenstates:
    loop_count = 0
    while not coarse_search_bounds(ith_energyLev):
        loop_count += 1
        if loop_count > 10000:
            print("Exceeded 10 000 energy estimates.")
            exit()

    end_sign = -end_sign

    #   Refine the energy by satisfying the right-boundary condition:
    loop_count = 0
    while not refine_bounds(ith_energyLev):
        loop_count += 1
        if loop_count > 10000:
            print("Exceeded 10 000 energy refining loops.")
            exit()

    #   Normalize the solution:
    psi /= pow(np.trapz(np.square(psi),dx=x_delta),0.5)

    #   Add units to energy:
    E_trial = (E_lowerBound + E_upperBound) / 2
    E_EigenEnergies[ith_energyLev] = energy_scale * E_trial

    #   Plot solution:
    plt.plot(region,psi)

    #   Reset loop:
    E_upperBound = E_trial
    E_lowerBound = E_trial

print(E_EigenEnergies)
plt.show()
