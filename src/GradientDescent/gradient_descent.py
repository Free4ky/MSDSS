import numpy as np
import numdifftools as nd
from src.utils import visualize


class GradientDescend:
    def __init__(self):
        self.n = None
        self.__norm2 = lambda x: np.sqrt(np.sum([i ** 2 for i in x]))

    def get_start_point_(self):
        return np.ones(self.n)

    def fit(self, func, n, e, h, iterations=25, verbose=True):
        k = 0
        self.n = n
        X_k = self.get_start_point_()
        history = [(*X_k, func(X_k))]
        for k in range(iterations):
            f_x = func(X_k)
            grad = nd.Gradient(func)(X_k)
            if self.__norm2(grad) <= e:
                return history
            if verbose:
                print(f'ITERATION {k + 1}\nx1 = {X_k[0]} \t\t x2 = {X_k[1]} \t y = {func(X_k)}')

            X_k = X_k - h * grad
            f_new = func(X_k)
            if f_new > f_x:
                h /= 2
            history.append((*X_k, func(X_k)))
        return history


if __name__ == '__main__':
    func = lambda x: x[0] ** 2 + 2 * x[1] ** 2 + np.exp(x[0] ** 2 + x[1] ** 2) - x[0] + 2 * x[1]
    gd = GradientDescend()
    history = gd.fit(
        func=func,
        n=2,
        e=0.01,
        h=0.05
    )
    visualize(history)
