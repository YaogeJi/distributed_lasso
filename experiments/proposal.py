import os
import numpy as np


exp_name = "proposal"
# batch_size_group = [5000, 500, 50, 10]
batch_size = 32
d = 20000
s = 9
sigma = np.sqrt(0.5)
m = 5
p = 1
rho = 0
# m_group = [1, 10, 100, 500]
# p_group = [1, 0.2, 0.05, 0.015]
# rho_group = [0, 0.9, 0.9, 0.9]
gamma_group = [100, 10, 1]
scheduler = "const"
max_iter = 20000
solver = "dual_average"
seed = 8989


for i, gamma in enumerate(gamma_group):
    params = "--batch_size {} -d {} -s {} --sigma {} -m {} -p {} -rho {} --max_iter {} --solver {}  --gamma {} --seed {} --scheduler {} --storing_filepath  ./output/{}/batch_size{}_d{}_s{}_sigma{}_seed{}/m{}_rho{}/ --storing_filename {}_gamma{}_{}.output".format(batch_size, d, s, sigma, m, p, rho, max_iter, solver, gamma, seed, scheduler, exp_name, batch_size, d, s, sigma, seed, m, rho, solver, gamma, scheduler)
    print('python main.py {}'.format(params))
    os.system('python main.py {}'.format(params))
