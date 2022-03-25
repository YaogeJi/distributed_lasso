import os


exp_name = "proposal"

N_group = [400,800,1200,1600,2000,2400,2800,3200]
d = 3200
s = 8
m = 20
p =   [1]
rho = [0]
gamma = 0.08789
total = 1

for i in range(len(rho_group)):
    p = p_group[i]
    rho = rho_group[i]
    for num_exp in range(total):
        command="-N {} -d {} -s {} --data_index {} --solver_mode pgd  --gamma {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename pgd_gamma{}.output".format(N, d, s, num_exp, gamma, exp_name, N, d, s, num_exp, gamma)
        print('python main.py {}'.format(command))
        os.system('python main.py {}'.format(command))

