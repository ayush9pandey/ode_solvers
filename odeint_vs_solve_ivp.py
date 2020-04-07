### This code is an attempt to benchmark the performance of two scipy.integrate tools: odeint and solve_ivp for commonly used minimal ODE models of biological or mechanical systems. ####
# A better benchmarking code for general ODEs is here : https://gist.github.com/Wrzlprmft/1d58ed1ec7f45049b922c849649f5cba
# More discussion here : https://github.com/scipy/scipy/issues/8257

from scipy.integrate import solve_ivp, odeint
from scipy.linalg import norm
import numpy as np 
from population_ode import population_ode
import matplotlib.pyplot as plt 
import time

def f1(t, x, fun = 0, *args):
    if fun == 0:
        # Toggle switch
        alpha1 = 1.3
        alpha2 = 1
        beta = 3
        gamma = 10
        pt_dot =  alpha1/(10 + x[1]**beta) - x[0]
        pl_dot = alpha2/(1 + x[0]**gamma) - x[1]
        y = np.array([pt_dot, pl_dot])
        return y
    if fun == 1:
        # Two member population control circuit
        y = population_ode(t, x)
        return y
    if fun == 2:
        # Repressilator ODEs
        beta = 20
        n = 2
        x_1, x_2, x_3 = x
        return np.array([beta / (1 + x_3**n) - x_1,
                     beta / (1 + x_1**n) - x_2,
                     beta / (1 + x_2**n) - x_3])
        return y
    if fun == 3:
        # Mass-action ODEs for gene expression
        k_bp, k_up, k_tx, k_br, k_ur, k_tl, k_be, k_ue, d_i, d, E_tot, P_tot, R_tot = [100, 10, 4, 10, 0.25, 2, 10, 0.5, 1, 1, 1000, 1000, 1000]
        x0, x1, x2, x3, x4, x5, x6, x7, x8 = x
        f0 = (k_bp + k_tx) * x2 - k_up * x0 * x1
        f1 = (k_bp + k_tx) * x2 - k_up * x0 * x1
        f2 = k_bp * x0 * x1 - (k_up + k_tx)*x2
        f3 = k_tx * x2 + k_ur * x5 - k_br * x3 * x4
        f4 = (k_ur + k_tl) * x5 - k_br * x3 * x4
        f5 = k_br * x3 * x4 - (k_ur + k_tl) * x5
        f6 = (k_ue + d_i) * x7 - k_be * x3 * x6
        f7 = k_be * x3 * x6 - (k_ue + d_i) * x7
        f8 = k_tl * x5 - d * x8
        y = [f0,f1,f2,f3,f4,f5,f6,f7,f8]
        return y
    if fun == 4:
        # Linear ODE
        y = x
        return y

n_funs = 5
list_of_timepoints = [np.linspace(0, 10, 10000), np.linspace(0, 100, 10), np.linspace(0, 10, 10000), np.linspace(0, 100, 100), np.linspace(0, 10, 1000)]
x0_pop =np.zeros(8)
x0_pop[6] = 100
x0_pop[7] = 200
x_init = np.zeros(9)
x_init[0] = 10
x_init[1] = 10000
x_init[4] = 10000
x_init[6] = 10000
list_of_init = [np.array([100, 2000]), x0_pop, np.array([1,1,1.2]), x_init, np.array([2]) ]
odeint_times = np.zeros(n_funs)
solve_ivp_times = np.zeros(n_funs)
error = np.zeros(n_funs)
for i in range(n_funs):
    timepoints = list_of_timepoints[i]
    x0 = list_of_init[i]
    t1 = time.time()
    sol1 = odeint(lambda x, t: f1(t, x, i), x0, timepoints)
    t2 = time.time()
    print('Solved using odeint for function {0} in time {1}'.format(i+1, t2-t1))
    odeint_times[i] = t2 - t1

    t1 = time.time()
    # if i == 3:
    #     solve_ivp_times[i] = np.nan
    #     continue
    sol2 = solve_ivp(lambda t, x: f1(t, x, i), (timepoints[0], timepoints[-1]), x0, t_eval = timepoints, method = 'LSODA')
    t2 = time.time()
    print('Solved using solve_ivp for function {0} in time {1}'.format(i+1, t2-t1))
    if np.shape(sol2.y)[1] != np.shape(timepoints)[0]:
        print('solve_ivp failed')
        solve_ivp_times[i] = np.nan
        continue
    solve_ivp_times[i] = t2 - t1
    plt.figure()
    plt.plot(timepoints, sol1, label = 'odeint')
    plt.plot(timepoints, sol2.y.T, label = 'solve_ivp')
    plt.legend()
    plt.savefig('Model ' + str(i+1))
    plt.close()
    error[i] = norm(sol1 - sol2.y.T)

print('Error in computations is', error)
model_num_list = [i+1 for i in range(n_funs)]
ax = plt.figure().gca()
plt.scatter(model_num_list, odeint_times, label = 'odeint times', linewidth = 3)
plt.scatter(model_num_list, solve_ivp_times, label = 'solve_ivp times', linewidth = 3)
plt.legend()
plt.ylabel('Time (in seconds)')
plt.xlabel('Different Model Examples')
plt.xticks(model_num_list)
plt.yticks
plt.savefig('odeint_vs_solve_ivp.png')
plt.show()

# Solve_ivp is extremely slow if time points are dense (large number of times)