import numpy as np
import time


def EOO(x, objfun, lb, ub, max_iter):
    row, col = x.shape[0], x.shape[1]
    f = objfun(x)
    fgbest = min(f)
    igbest = np.where(min(f) == fgbest)
    xbest = x[igbest, :]
    fbst = np.zeros((1, max_iter))

    ct = time.time()
    it = 0
    while it < max_iter:
        for i in range(row):
            L = np.random.uniform(3, 5)
            T = ((L - 3) / (5 - 3) * 10) - 5
            if i > 1:
                E = ((i-1) / (row - 1)) - 0.5 # linearly decreased from 0.5 to −0.5
            else:
                E = - 0.5
            C = (((L - 3)/(5 - 3)) * 2)+0.6
            Y = T+E+L+np.random.rand() * (xbest - x[i, :]) #  final energy of EO in each solution (iteration)
            x[i, :] = x[i, :] * C
        f = objfun(x)

        # Find global best and local best
        minf = min(f)
        iminf = np.where(min(f) == minf)
        if minf <= fgbest:
            fgbest = minf
            bestsol = x[iminf, :]
            fbst[it] = minf
        else:
            fbst[it] = fgbest
            bestsol = xbest
    ct = time.time() - ct

    return fgbest, fbst, bestsol, ct



