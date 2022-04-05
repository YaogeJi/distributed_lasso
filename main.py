import time
import argparse
import pickle
import os
from generator import Generator
from network import ErodoRenyi
from solver import *
from scheduler import *

# configuration
parser = argparse.ArgumentParser(description='distributed optimization')
parser.add_argument('--storing_filepath', default='', type=str, help='storing_file_path')
parser.add_argument('--storing_filename', default='', type=str, help='storing_file_name')
## data
parser.add_argument("-N", "--num_samples", type=int)
parser.add_argument("-d", "--num_dimensions", type=int)
parser.add_argument("-s", "--sparsity", type=int)
parser.add_argument("-k", type=float, default=0.25)
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--data_index", type=int, default=0)
## network
parser.add_argument("-m", "--num_nodes", default=1, type=int)
parser.add_argument("-p", "--probability", default=1, type=float)
parser.add_argument("-rho", "--connectivity", default=0, type=float)
## solver
parser.add_argument("--solver", choices=("pgd", "pcta", "patc","netlasso","primaldual","pgextra"))
parser.add_argument("--projecting", action="store_true")
parser.add_argument("--max_iter", type=int, default=1000000)
parser.add_argument("--tol", type=float, default=1e-8)
parser.add_argument("--iter_type", choices=("lagrangian", "projected"))
parser.add_argument("--gamma", type=float)
parser.add_argument("--beta", type=float, default=0)
parser.add_argument("--communication", type=int, default=1)
parser.add_argument("--local_computation", type=int, default=1)
parser.add_argument("--scheduler", choices=("const","diminish"), default="const")
parser.add_argument("--betascheduler", choices=("const","diminish"), default="const")
## others
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


def main():
    # preprocessing data
    data_path = "../data/N{}_d{}_s{}_k{}_sigma{}/".format(
        args.num_samples, args.num_dimensions, args.sparsity, args.k, args.sigma)
    data_file = data_path + "exp{}.data".format(args.data_index)
    network_path = "../network/"
    network_file = network_path + "m{}_rho{}.network".format(args.num_nodes, args.connectivity)

    ## processing data
    try:
        X, Y, ground_truth, optimal_lambda, min_stat_error = pickle.load(open(data_file, "rb"))
        
    except FileNotFoundError:
        os.makedirs(data_path, exist_ok=True)
        generator = Generator(args.num_samples, args.num_dimensions, args.sparsity, args.k, args.sigma)
        X, Y, ground_truth, optimal_lambda, min_stat_error = generator.generate()
        pickle.dump([X, Y, ground_truth, optimal_lambda, min_stat_error], open(data_file, "wb"))
    r = np.linalg.norm(ground_truth, ord=1)
    ## processing network
    try:
        w = pickle.load(open(network_file, "rb"))
    except:
        w = ErodoRenyi(m=args.num_nodes, rho=args.connectivity, p=args.probability).generate()
        os.makedirs(network_path, exist_ok=True)
        pickle.dump(w, open(network_file, "wb"))
    print(np.sum(X), np.sum(Y))
    w = (w + np.eye(w.shape[0])) / 2 
    ## process stepsize
    if args.scheduler == "const":
        gamma = ConstScheduler(args.gamma)
    elif args.scheduler == "diminish":
        gamma = DiminishScheduler(args.gamma)

    # solver run
    if args.solver == 'pgd':
        print("PGD")
        solver = PGD(args.max_iter, gamma, r)
    elif args.solver == 'pcta':
        print("projected_cta")
        solver = PCTA(args.max_iter, gamma, r, w, args.communication, args.local_computation)
    elif args.solver == 'patc':
        print("projected_atc")
        solver = PATC(args.max_iter, gamma, r, w, args.communication, args.local_computation)
    elif args.solver == 'netlasso':
        print("netlasso")
        solver = NetLasso(args.max_iter, gamma, r, w, args.communication, args.local_computation)
    elif args.solver == 'pgextra':
        print("pgextra")
        solver = PGExtra(args.max_iter, gamma, r, w, args.communication, args.local_computation)
    elif args.solver == 'primaldual':
        if args.betascheduler == "const":
            beta = ConstScheduler(args.beta)
        elif args.betascheduler == "diminish":
            beta = DiminishScheduler(args.beta)
        print("primal_dual")
        solver = PrimalDual(args.max_iter, gamma, beta, r, w, args.communication, args.local_computation)
    else:
        raise NotImplementedError("solver mode currently only support centralized or distributed")
    start_timer = time.time()
    outputs = solver.fit(X, Y, ground_truth, verbose=args.verbose)
    finish_timer = time.time()
    print("solver spend {} seconds".format(finish_timer-start_timer))
    output_filepath = args.storing_filepath
    output_filename = args.storing_filename
    os.makedirs(output_filepath, exist_ok=True)
    pickle.dump(outputs, open(output_filepath + output_filename, "wb"))


if __name__ == "__main__":
    main()
