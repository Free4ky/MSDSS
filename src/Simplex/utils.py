import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Qt5Agg')


def func(X):
    '''

    :param X: coordinate in feature space
    :return: y
    '''
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    x1, x2 = X[:, 0], X[:, 1]
    return x1 ** 2 + 2 * x2 ** 2 + np.exp(x1 ** 2 + x2 ** 2) - x1 + 2 * x2


def func1(X1, X2):
    return X1 ** 2 + 2 * X2 + np.exp(X1 ** 2 + X2 ** 2) - X1 + 2 * X2


def visualize(history):
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)

    X1, X2 = np.meshgrid(x1, x2)

    Z = func1(X1, X2)

    fig = plt.figure(figsize=plt.figaspect(2.))
    ax = fig.add_subplot(2, 1, 1)
    _, _, Z_P = zip(*history)
    ax.plot(Z_P)
    ax.set_title('Optimization process')
    ax.set_ylabel('f(X)')
    ax.set_xlabel('X')
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    x1_p, x2_p, z_p = zip(*history)
    ax.plot(x1_p, x2_p, z_p, color='red')
    ax.set_title('3d plot')
    plt.show()
