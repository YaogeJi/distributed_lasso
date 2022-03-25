import os


exp_name = "ball_verification"

# N_group = [400,800,1200,1600,2000,2400,2800,3200]
N_group = [400]
d = 3200
s = 8
m = 20
p =   0.7
rho = 0.9
solver = "primaldual"
gamma = 0.0001
scheduler = "const"
beta = 0.005
betascheduler = "const"
max_iter = 100000
total = 1

for N in N_group:
    for num_exp in range(total):
        print(N, num_exp)
        if solver == "primaldual":
            command="-N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --max_iter {} --solver {}  --gamma {} --scheduler {} --beta {} --betascheduler {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename {}_m{}_rho{}_gamma{}_{}_beta{}_{}.output".format(N, d, s, m, p, rho, num_exp,max_iter, solver, gamma, scheduler, beta, betascheduler, exp_name, N, d, s, num_exp, solver, m, rho, gamma, scheduler, beta, betascheduler)
        else:
            command="-N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --max_iter {} --solver {}  --gamma {} --scheduler {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename {}_m{}_rho{}_gamma{}_{}.output".format(N, d, s, m, p, rho, num_exp,max_iter, solver, gamma, scheduler, exp_name, N, d, s, num_exp, solver, m, rho, gamma, scheduler)
        print('python main.py {}'.format(command))
        os.system('python main.py {}'.format(command))

