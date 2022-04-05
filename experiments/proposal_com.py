import os
import numpy as np
import math


exp_name = "proposal"

#N_group = [1200,1600,2000,2400,2800,3200]
N_group = [240, 320, 420,  520,  560,   680,   860]
d_group = [400, 800, 3200, 6400, 12800, 25600, 51200]
s_group = [5,   6,   8,    8,    9,     10,    10]
m = 20
p =   0.7
rho = 0.9
solver_group = ["patc", "netlasso", "primaldual"]
gamma_group = [0.08789,0.19,0.001]
scheduler = "const"
communication_group = [6, 8, 10, 11, 14, 15, 17]
beta = 0.1
betascheduler = "const"
max_iter_group = [500, 500, 8000]
num_exp = 0

for i,N in enumerate(N_group):
    d = d_group[i]
    s = s_group[i]
    communication = communication_group[i]
    for j,solver in enumerate(solver_group):
        gamma = gamma_group[j]
        max_iter = max_iter_group[j]
        print(N, solver)
        if solver == "primaldual":
            command="-N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --max_iter {} --solver {}  --gamma {} --scheduler {} --beta {} --betascheduler {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename {}_m{}_rho{}_gamma{}_{}_beta{}_{}.output".format(N, d, s, m, p, rho, num_exp,max_iter, solver, gamma, scheduler, beta, betascheduler, exp_name, N, d, s, num_exp, solver, m, rho, gamma, scheduler, beta, betascheduler)
        elif solver == "patc":
            command="-N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --max_iter {} --solver {}  --gamma {} --communication {} --scheduler {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename {}_m{}_rho{}_gamma{}_communication{}_{}.output".format(N, d, s, m, p, rho, num_exp,max_iter, solver, gamma, communication, scheduler, exp_name, N, d, s, num_exp, solver, m, rho, gamma, communication, scheduler)
        else:
            command="-N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --max_iter {} --solver {}  --gamma {} --scheduler {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename {}_m{}_rho{}_gamma{}_{}.output".format(N, d, s, m, p, rho, num_exp, max_iter, solver, gamma, scheduler, exp_name, N, d, s, num_exp, solver, m, rho, gamma, scheduler)
        print('python main.py {}'.format(command))
        os.system('python main.py {}'.format(command))

