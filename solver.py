import numpy as np
import time
from projection import proj_l1ball as proj
from sklearn import linear_model
import scipy

class Solver:
    def __init__(self, max_iteration, gamma):
        self.max_iteration = int(max_iteration)
        self.gamma = gamma

    def fit(self, X, Y, ground_truth, verbose):
        raise NotImplementedError

    def show_param(self):
        raise NotImplementedError


class PGD(Solver):
    def __init__(self, max_iteration, gamma, project_radius):
        super(PGD, self).__init__(max_iteration, gamma)
        self.r = project_radius

    def iterate(self, theta, x, y):
        N, d = x.shape
        gamma = self.gamma()
        theta = theta - gamma / N * (theta @ x.T - y.T) @ x
        theta = proj(theta, self.r)
        return theta

    def fit(self, X, Y, ground_truth, verbose):
        # initialize parameters we need
        loss = []
        N, d = X.shape
        # initialize iterates
        theta = 0.0 * np.ones((1, d))
        # iterates!
        loss_matrix = []
        for step in range(self.max_iteration):
            # theta_last = theta.copy()
            theta = self.iterate(theta, X, Y)
            if ground_truth is not None:
                loss_matrix.append(np.linalg.norm(theta - ground_truth.T, ord=2) ** 2)
                if step % 100 == 0:
                    print(step, loss_matrix[-1])
            # if np.linalg.norm(theta-theta_last, ord=2) < self.terminate_condition:
            # if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition:
            # if np.linalg.norm(theta - theta_last, ord=2) / np.linalg.norm(theta_last, ord=2)  < self.terminate_condition:
            #     print("Early convergence at step {} with log loss {}, I quit.".format(step, log_loss[-1]))
            #     return theta, log_loss
        else:
            print("Max iteration, I quit.")
            return theta, loss_matrix


class PCTA(PGD):
    def __init__(self, max_iteration, gamma, project_radius, w, communication=1, local_computation=1):
        super(PCTA, self).__init__(max_iteration, gamma, project_radius)
        self.w = w
        self.m = self.w.shape[0]
        self.communication = communication
        self.local_computation = local_computation

    def iterate(self, theta, x, y):
        m, n, d = x.shape
        gamma = self.gamma()
        theta = np.linalg.matrix_power(self.w, self.communication) @ theta.squeeze(axis=2)
        theta = np.expand_dims(theta, axis=2)
        for local_updates in range(self.local_computation):
            theta = theta - gamma / n * x.transpose(0,2,1) @ (x @ theta - y)
            theta = theta.squeeze(axis=2)
            theta = (proj(theta, self.r)).reshape(m, d, 1)
        return theta

    def fit(self, X, Y, ground_truth, verbose):
        # Initialize parameters we need
        N, d = X.shape
        assert N % self.m == 0, "sample size {} is indivisible by {} nodes.".format(N, self.m)
        # Initialize iterates
        theta = 1.0 * np.ones((self.m, d, 1))
        x = X.reshape(self.m, int(N / self.m), d)
        y = Y.reshape(self.m, int(N / self.m), 1)

        # iterates!
        loss_matrix = []
        for step in range(self.max_iteration):
            if verbose:
                if step % 100 == 0 and step != 0:
                    if ground_truth is not None:
                        print("{}/{}, loss = {}".format(step, self.max_iteration, loss_matrix[-1]))
                    else:
                        print("{}/{}".format(step, self.max_iteration))
            if ground_truth is not None:
                "optimization error has bug here. shape of comparison is not consensus."
                loss_matrix.append(np.linalg.norm(theta.squeeze() - np.repeat(ground_truth.T, self.m, axis=0), ord=2) ** 2 / self.m)
            # theta_last = theta.copy()
            theta = self.iterate(theta, x, y)
            # if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition:
            # if np.linalg.norm(theta - theta_last, ord=2) / np.linalg.norm(theta_last, ord=2) < self.terminate_condition:

            # if np.max(np.linalg.norm(theta-theta_last, ord=2, axis=1)) < self.terminate_condition:
            #     print("Early convergence at step {} with log loss {}, I quit.".format(step, log_loss[-1]))
            #     return theta, log_loss
        else:
            print("Max iteration, I quit.")
            return theta, loss_matrix


class PATC(PCTA):
    def __init__(self, max_iteration, gamma, project_radius, w, communication=1, local_computation=1):
        super(PATC, self).__init__(max_iteration, gamma, project_radius, w, communication, local_computation)

    def iterate(self, theta, x, y):
        m, n, d = x.shape
        gamma = self.gamma()
        for local_updates in range(self.local_computation):
            theta = theta - gamma / n * x.transpose(0,2,1) @ (x @ theta - y)
        theta = np.linalg.matrix_power(self.w, self.communication) @ theta.squeeze(axis=2)
        theta = (proj(theta, self.r)).reshape(self.m,d,1)
        return theta
        # iterates!


class NetLasso(PCTA):
    def __init__(self, max_iteration, gamma, project_radius, w, communication=1, local_computation=1):
        super(NetLasso, self).__init__(max_iteration, gamma, project_radius, w, communication, local_computation)
        if self.local_computation != 1:
            raise NotImplementedError("multi local updates not implemented")
        self.w = w
        self.m = self.w.shape[0]
    
    def iterate(self, theta, x, y):
        m, n, d = x.shape
        gamma = self.gamma()
        theta = np.linalg.matrix_power(self.w, self.communication) @ theta.squeeze(axis=2)
        theta = np.expand_dims(theta, axis=2)
        grad_now = gamma / n * x.transpose(0,2,1) @ (x @ theta - y)
        last_grad = self.last_grad
        self.last_grad = grad_now
        grad_track = self.grad_track
        grad_track = np.linalg.matrix_power(self.w, self.communication) @ (grad_track + grad_now - last_grad).squeeze(axis=2)
        grad_track = np.expand_dims(grad_track, axis=2)
        theta -= gamma * grad_track
        theta = (proj(theta.squeeze(axis=2), self.r)).reshape(self.m,d,1)
        self.grad_track = grad_track
        return theta

    def fit(self, X, Y, ground_truth, verbose):
        N, d = X.shape
        self.grad_track = np.zeros((self.m, d, 1))
        self.last_grad = np.zeros((self.m, d, 1))
        theta, loss_matrix = super().fit(X, Y, ground_truth, verbose)
        return theta, loss_matrix


class PrimalDual(PCTA):
    def __init__(self, max_iteration, gamma, beta, project_radius, w, communication=1, local_computation=1):
        super(PrimalDual, self).__init__(max_iteration, gamma, project_radius, w, communication, local_computation)
        self.beta = beta

    def iterate(self, theta, x, y):
        m, n, d = x.shape
        gamma = self.gamma()
        beta = self.beta()
        theta = np.expand_dims((self.w + np.eye(m)) / 2 @ theta.squeeze(axis=2),axis=2) - gamma / n * x.transpose(0,2,1) @ (x @ theta - y) - self.dual_var
        self.dual_var += beta * (theta - np.expand_dims((self.w + np.eye(m)) / 2 @ theta.squeeze(axis=2), axis=2))
        theta = (proj(theta.squeeze(axis=2), self.r)).reshape(self.m,d,1)
        return theta
        # iterates!
    
    def fit(self, X, Y, ground_truth, verbose):
        N, d = X.shape
        self.dual_var = 0.0 * np.ones((self.m, d, 1))
        theta, loss_matrix = super().fit(X, Y, ground_truth, verbose)
        return theta, loss_matrix


class PGExtra(PCTA):        
    def iterate(self, theta, x, y):
        m, n, d = x.shape
        gamma = self.gamma()
        theta = np.expand_dims((self.w + np.eye(m)) / 2 @ theta.squeeze(axis=2),axis=2) - gamma / n * x.transpose(0,2,1) @ (x @ theta - y) - self.dual_var
        self.dual_var += 0.5 * (theta - np.expand_dims(self.w @ theta.squeeze(axis=2), axis=2))
        theta = (proj(theta.squeeze(axis=2), self.r)).reshape(self.m,d,1)
        return theta
        # iterates!
    
    def fit(self, X, Y, ground_truth, verbose):
        N, d = X.shape
        self.dual_var = 0.0 * np.ones((self.m, d, 1))
        theta, loss_matrix = super().fit(X, Y, ground_truth, verbose)
        return theta, loss_matrix
