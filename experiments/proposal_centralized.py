import os


exp_name = "proposal"

N_group = [500,1000,2000,4000,8000,12000,16000]
d = 20000
s = 9
gamma = 0.08789
total = 1
max_iter = 2000

for N in N_group:
    for num_exp in range(total):
        command="-N {} -d {} -s {} --max_iter {} --data_index {} --solver pgd  --gamma {} --verbose --storing_filepath ./output/{}/N{}_d{}_s{}_exp{}/ --storing_filename pgd_gamma{}.output".format(N, d, s, max_iter ,num_exp, gamma, exp_name, N, d, s, num_exp, gamma)
        print('python main.py {}'.format(command))
        os.system('python main.py {}'.format(command))

