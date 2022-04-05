import os
import numpy as np
import math


exp_name = "proposal"

#N_group = [1200,1600,2000,2400,2800,3200]
N_group = [500,1000,2000,4000,8000,12000,16000]
d = 20000
s = 9
m = 20
p =   0.7
rho = 0.9
solver = "patc"
gamma = 0.08789
scheduler = "const"
communication = 15
#beta = 0.1
#betascheduler = "const"
max_iter = 2000
total = 1

for N in N_group:
    for num_exp in range(total):
        print(N, num_exp)
        if solver == "primaldual":
            command="-N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --max_iter {} --solver {}  --gamma {} --scheduler {} --beta {} --betascheduler {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename {}_m{}_rho{}_gamma{}_{}_beta{}_{}.output".format(N, d, s, m, p, rho, num_exp,max_iter, solver, gamma, scheduler, beta, betascheduler, exp_name, N, d, s, num_exp, solver, m, rho, gamma, scheduler, beta, betascheduler)
        else:
            if communication is not None or communication != 1:
                command="-N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --max_iter {} --solver {}  --gamma {} --communication {} --scheduler {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename {}_m{}_rho{}_gamma{}_communication{}_{}.output".format(N, d, s, m, p, rho, num_exp,max_iter, solver, gamma, communication, scheduler, exp_name, N, d, s, num_exp, solver, m, rho, gamma, communication, scheduler)
            else:
                command="-N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --max_iter {} --solver {}  --gamma {} --scheduler {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename {}_m{}_rho{}_gamma{}_{}.output".format(N, d, s, m, p, rho, num_exp, max_iter, solver, gamma, scheduler, exp_name, N, d, s, num_exp, solver, m, rho, gamma, scheduler)
        print('python main.py {}'.format(command))
        os.system('python main.py {}'.format(command))

