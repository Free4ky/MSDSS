from src.utils import visualize
# import numpy as np
import numdifftools as nd
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian


class Newton:
    def __init__(self):
        self.n = None
        self.__norm2 = lambda x: np.sqrt(np.sum([i ** 2 for i in x]))

    def get_start_point_(self):
        return np.ones(self.n)

    def __is_positive(self, X):
        '''

        :param X: matrix
        :return:
        '''

        if X.shape[0] != X.shape[1]:
            return False

        for i in range(1, X.shape[0] + 1):
            minor = X[:i, :i]
            determinant = np.linalg.det(minor)
            if determinant <= 0:
                return False

        return True

    def fit(self, func, n, e, iterations=100, verbose=True):
        k = 0
        self.n = n
        X_k = self.get_start_point_()
        history = [(*X_k, func(X_k))]
        for k in range(iterations):
            f_x = func(X_k)
            grad = nd.Gradient(func)(X_k)
            if self.__norm2(grad) <= e:
                return history
            hessian_k = jacobian(egrad(func))(X_k)
            p_k = - np.linalg.inv(hessian_k) @ grad
            if verbose:
                print(f'ITERATION {k + 1}\nx1 = {X_k[0]} \t\t x2 = {X_k[1]} \t y = {func(X_k)}')
            if self.__is_positive(hessian_k):
                X_k = X_k + p_k
            else:
                h_k = (grad @ p_k) / ((hessian_k @ p_k) @ grad)
                X_k = X_k - h_k * grad
            history.append((*X_k, func(X_k)))
        return history


if __name__ == '__main__':
    func = lambda x: x[0] ** 2 + 2 * x[1] ** 2 + np.exp(x[0] ** 2 + x[1] ** 2) - x[0] + 2 * x[1]
    newton = Newton()
    history = newton.fit(
        func,
        n=2,
        e=0.01,
    )
    visualize(history)

    # g = np.array(nd.Gradient(func)(np.ones(2)))
    # h = np.vstack((g, g[::-1]))
    # print(nd.Gradient(func)(h))
