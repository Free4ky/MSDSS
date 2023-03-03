from copy import deepcopy
import operator
import numpy.random
import pandas as pd

from src.Simplex.utils import *

numpy.random.seed(42)


class Simplex:
    def __init__(self):
        self.dim = None
        self.eps = None
        self.m = None
        self.func = None

    def get_start_point_(self):
        return np.ones(self.dim)

    def sigma1_(self):
        return ((np.sqrt(self.dim + 1) - 1) * self.m) / (self.dim * np.sqrt(2))

    def sigma2_(self):
        return (np.sqrt(self.dim + 1) + self.dim - 1) / (self.dim * np.sqrt(2)) * self.m

    def get_other_points(self, start_point):
        result = deepcopy(start_point)
        for i in range(self.dim):
            point = np.zeros(self.dim)
            # iterate through coordinate
            for j, c in enumerate(start_point):
                if i == j:
                    point[j] = c + self.sigma1_()
                    # print('SIGMA1', self.sigma1_())
                else:
                    point[j] = c + self.sigma2_()
                    # print('SIGMA2 ', self.sigma2_())
            result = np.vstack((result, point))
        return result

    def except_i_(self, X, i):
        l = X[:i]
        r = X[i + 1:]
        return np.concatenate((l, r), axis=0)

    def calculate_gravity_center_(self, points, max_idx):
        excepted_i = self.except_i_(points, max_idx)
        res = np.sum(excepted_i, axis=1)
        return res / self.dim

    def get_min_(self, X):
        return min(enumerate(func(X)), key=operator.itemgetter(1))

    def reduction_(self, X):
        '''

        :param X: old simplex
        :return: new simplex after reduction has been performed
        '''
        idx_min, _ = self.get_min_(X)
        for i in range(len(X)):
            if i != idx_min:
                X[i] = X[idx_min] + 0.5 * (X[i] - X[idx_min])
        return X

    def create_df_(self, X):
        columns = list(map(lambda x: str(x), range(X.shape[1])))
        columns.append('f(x)')
        return pd.DataFrame(
            columns=columns,
            index=list(range(X.shape[0])),
            data=np.c_[X, self.func(X)]
        )

    def fit(self, func, dim, eps, m, iterations=500, verbose=True):
        history = []
        self.eps = eps
        self.dim = dim
        self.m = m
        self.func = func
        X = self.get_start_point_()
        # print('Start point', X, sep='\n')
        # print(start_point)
        # other_points = list(self.get_other_points(start_point))
        X = self.get_other_points(X)

        # print('ALL', X, sep='\n')

        f_in_x_c = np.inf
        i = 0
        while i < iterations:
            idx_max, y_max = max(enumerate(func(X)), key=operator.itemgetter(1))
            gravity_center = self.calculate_gravity_center_(X, idx_max)
            reflected_point = 2 * gravity_center - X[idx_max]
            y_reflected = func(reflected_point)
            if y_reflected < y_max:
                X[idx_max] = reflected_point
                y_max = y_reflected
            else:
                # reduction
                X = self.reduction_(X)

                # calculate simplex gravity center
            x_c = 1 / (self.dim + 1) * np.sum(X, axis=1)
            f_in_x_c = func(x_c)
            i += 1
            idx_min, y_min = self.get_min_(X)
            history.append((*X[idx_min], y_min))
            if verbose:
                print(f'\nITERATION {i + 1}')
            print(self.create_df_(X))

            if all(np.abs(i - f_in_x_c) < eps for i in func(X)):
                return history

        return history


if __name__ == '__main__':
    simplex = Simplex()
    history = simplex.fit(
        func,
        dim=2,
        eps=0.1,
        m=0.5,
    )
    visualize(history)
    # print(history[-1])
