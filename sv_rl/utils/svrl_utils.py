import torch
import numpy as np
import math, time

import cvxpy as cvx
from fancyimpute import SoftImpute, BiScaler


def softimp(inputs, h, w, mask_prob):
    # transpose to ensure h < w
    # inputs = inputs.cpu().numpy().T
    inputs = inputs.cpu().numpy()
    subtract_val = (inputs.max() + inputs.min()) / 2.
    divide_val = (inputs.max() - inputs.min()) / 2. + 1e-7
    inputs = (inputs - subtract_val) / divide_val

    mask = np.random.binomial(1, mask_prob, h * w).reshape(h, w).astype(float)
    mask[mask < 1] = np.nan
    W = SoftImpute(verbose=False).fit_transform(mask * inputs)
    W[W < -1] = -1
    W[W > 1] = 1
    est_matrix = W * divide_val + subtract_val

    return torch.from_numpy(est_matrix)


def nuclear_norm_solve(A, mask, mu):
    X = cvx.Variable(shape=A.shape)
    objective = cvx.Minimize(mu * cvx.norm(X, "nuc") + cvx.sum_squares(cvx.multiply(mask, X-A)))
    problem = cvx.Problem(objective, [])
    problem.solve(solver=cvx.SCS)
    return X.value


def nucnorm(inputs, h, w, mask_prob):
    inputs = inputs.cpu().numpy()
    subtract_val = (inputs.max() + inputs.min()) / 2.
    divide_val = (inputs.max() - inputs.min()) / 2. + 1e-7
    inputs = (inputs - subtract_val) / divide_val
    mask = np.random.binomial(1, mask_prob, h * w).reshape(h, w)
    W = nuclear_norm_solve(inputs, mask, mu=3)
    W[W < -1] = -1
    W[W > 1] = 1
    est_matrix = W * divide_val + subtract_val

    return torch.from_numpy(est_matrix)


def bregman(image, mask, weight, eps=1e-3, max_iter=100):
    rows, cols, dims = image.shape
    rows2 = rows + 2
    cols2 = cols + 2
    total = rows * cols * dims
    shape_ext = (rows2, cols2, dims)

    u = np.zeros(shape_ext)
    dx = np.zeros(shape_ext)
    dy = np.zeros(shape_ext)
    bx = np.zeros(shape_ext)
    by = np.zeros(shape_ext)

    u[1:-1, 1:-1] = image
    u[0, 1:-1] = image[1, :]
    u[1:-1, 0] = image[:, 1]
    u[-1, 1:-1] = image[-2, :]
    u[1:-1, -1] = image[:, -2]

    i = 0
    rmse = np.inf
    lam = 2 * weight
    norm = (weight + 4 * lam)

    while i < max_iter and rmse > eps:
        rmse = 0
        for k in range(dims):
            for r in range(1, rows + 1):
                for c in range(1, cols + 1):
                    uprev = u[r, c, k]

                    ux = u[r, c + 1, k] - uprev
                    uy = u[r + 1, c, k] - uprev

                    if mask[r - 1, c - 1]:
                        unew = (lam * (u[r + 1, c, k] +
                                       u[r - 1, c, k] +
                                       u[r, c + 1, k] +
                                       u[r, c - 1, k] +
                                       dx[r, c - 1, k] -
                                       dx[r, c, k] +
                                       dy[r - 1, c, k] -
                                       dy[r, c, k] -
                                       bx[r, c - 1, k] +
                                       bx[r, c, k] -
                                       by[r - 1, c, k] +
                                       by[r, c, k]
                                       ) + weight * image[r - 1, c - 1, k]
                                ) / norm
                    else:
                        unew = (u[r + 1, c, k] +
                                u[r - 1, c, k] +
                                u[r, c + 1, k] +
                                u[r, c - 1, k] +
                                dx[r, c - 1, k] -
                                dx[r, c, k] +
                                dy[r - 1, c, k] -
                                dy[r, c, k] -
                                bx[r, c - 1, k] +
                                bx[r, c, k] -
                                by[r - 1, c, k] +
                                by[r, c, k]
                                ) / 4.0
                    u[r, c, k] = unew
                    rmse += (unew - uprev) ** 2

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    s = ux + bxx
                    if s > 1 / lam:
                        dxx = s - 1 / lam
                    elif s < -1 / lam:
                        dxx = s + 1 / lam
                    else:
                        dxx = 0
                    s = uy + byy
                    if s > 1 / lam:
                        dyy = s - 1 / lam
                    elif s < -1 / lam:
                        dyy = s + 1 / lam
                    else:
                        dyy = 0

                    dx[r, c, k] = dxx
                    dy[r, c, k] = dyy
                    bx[r, c, k] += ux - dxx
                    by[r, c, k] += uy - dyy

        rmse = np.sqrt(rmse / total)
        i += 1
    return np.squeeze(np.asarray(u[1:-1, 1:-1]))


def tvm(input_array, h, w, keep_prob=0.8, lambda_tv=0.03):
    # expand input arr to 3-dims
    input_array = input_array.unsqueeze(2).cpu().numpy()
    mask = np.random.uniform(size=input_array.shape[:2])
    mask = mask < keep_prob
    return torch.from_numpy(bregman(input_array, mask, weight=2.0/lambda_tv))
