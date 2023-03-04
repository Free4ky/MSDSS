from src.utils import *

machineAcc = 0.000000001


class HookJeeves:
    def __init__(self):
        self.dim = None
        self.e = None
        self.dim = None

    def get_start_point_(self):
        return np.ones(self.dim)

    def exploring_search_(self, x, h, func):
        x_res = x[:]
        fb = func(x_res)
        for i in range(0, len(x_res)):
            bn = x_res
            bn[i] = bn[i] + h[i]
            fc = func(bn)
            if (fc + machineAcc < fb):
                x_res = bn
                fb = fc
            else:
                bn[i] = bn[i] - 2 * h[i]
                fc = func(bn)
                if (fc + machineAcc < fb):
                    x_res = bn
                    fb = fc
        return x_res

    def fit(self, dim, h: list, d, m, e, func):
        assert dim == len(h)
        assert d > 1
        assert m > 0
        self.dim = dim
        self.d = d
        self.m = m
        self.e = e
        Path1 = []
        Path2 = []
        Path3 = []
        Path4 = []

        b1 = self.get_start_point_()
        history = []
        history.append((*b1, *func(b1)))

        runOuterLoop = True
        while (runOuterLoop):
            runOuterLoop = False
            runInnerLoop = True
            xk = b1  # step1
            b2 = self.exploring_search_(b1, h, func)  # step2
            Path1.append(b1)
            Path2.append(b2)
            Path3.append(xk)
            while (runInnerLoop):
                Path1.append(b1)
                Path2.append(b2)
                runInnerLoop = False
                for i in range(len(b1)):  # step3
                    xk[i] = b1[i] + m * (b2[i] - b1[i])
                Path3.append(xk)
                x = self.exploring_search_(xk, h, func)  # step4
                Path4.append(x)
                b1 = b2  # step5
                fx = func(x)
                fb1 = func(b1)
                if (fx + machineAcc < fb1):  # step6
                    b2 = x
                    runInnerLoop = True  # to step3
                elif (fx - machineAcc > fb1):  # step7
                    runOuterLoop = True  # to step1
                    break
                else:
                    s = 0
                    for i in range(len(h)):
                        s += h[i] * h[i]
                    if (e * e + machineAcc > s):  # step8
                        break  # to step10
                    else:
                        for i in range(len(h)):  # step9
                            h[i] = h[i] / d
                        runOuterLoop = True  # to step1
            history.append((*b1, *func(b1)))
        return b1, history  # step10


if __name__ == '__main__':
    hj = HookJeeves()
    res, history = hj.fit(
        dim=2,
        h=[0.2, 0.2],
        d=2,
        e=0.01,
        m=2,
        func=func
    )
    visualize(history)
