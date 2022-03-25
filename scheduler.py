import numpy as np


class Scheduler(object):
    def __init__(self, start_stepsize):
        self.start_stepsize = start_stepsize
        self.count = 0
    def __call__(self, iteration):
        self.count += 1


class ConstScheduler(Scheduler):
    def __init__(self, start_stepsize):
        super(ConstScheduler, self).__init__(start_stepsize)

    def __call__(self, iteration=None):
        super(ConstScheduler, self).__call__(iteration)
        return self.start_stepsize


class DiminishScheduler(Scheduler):
    def __init__(self, start_stepsize=1):
        super(DiminishScheduler, self).__init__(start_stepsize)

    def __call__(self, iteration=None):
        if iteration is None:
            iteration = self.count
        super(DiminishScheduler, self).__call__(iteration)
        return self.start_stepsize / (iteration + 1)


if __name__ == "__main__":
    gamma_const = ConstScheduler(0.01)
    gamma_diminishing = DiminishScheduler()
    for it in range(100):
        print("This is iteration {}".format(it))
        print(gamma_const())
        print(gamma_diminishing())
