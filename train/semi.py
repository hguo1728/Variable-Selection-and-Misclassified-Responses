import numpy as np
import time
import pycasso
from scipy.optimize import minimize
import subprocess
from sklearn.decomposition import KernelPCA, PCA
import json
import os

import func_timeout
from datetime import datetime

from train.utils import info_stat, glm_fun, glm_grad
from train import path_following

def semi(args):
    
    n = args.n
    p = args.p
    n_v = args.n_v
    Z = args.Z
    Z_val = args.Z_val
    Y_star = args.Y_star
    Y_val = args.Y_val
    Y_star_val = args.Y_star_val

    discrete_idx = args.discrete_idx


    # calculate bandwidth & smoothing parameter

    n_bandwidth = 5
    bandwidth = np.linspace(0.5 * (n_v ** (- 1 / (4 + p))), 2 * n_v ** (- 1 / (4 + p)), num = n_bandwidth)
    w = 1 * n_v ** (- 2 / (4 + p))


    N = args.N_iter if args.use_intercept else 1

    Criterion_betas = np.zeros((n_bandwidth, N, p))
    Criterion_lambdas = np.zeros((n_bandwidth, N))
    Criterion_values = np.ones((n_bandwidth, N)) * 1e10
    Returned_intercepts = np.zeros((n_bandwidth, N))


    ############################## initialize ##############################
    beta_old = np.zeros(p)
    intercept_old = 0
    
    model = pycasso.Solver(Z, Y_star, lambdas=(100, 0.0015), family='binomial',
                        penalty=args.penalty, gamma=3.7, useintercept=args.use_intercept, prec=0.0001, max_ite=1000, verbose=False)
    model.train()


    stat_ = np.zeros(100)
    for j in range(100):
        beta_ = model.coef()['beta'][j]
        intercept_ = model.coef()['intercept'][j] if args.use_intercept else 0
        lambda_ = model.lambdas[j]
        stat_[j] = info_stat(args, lambda_, beta_, intercept_)
    
    idx_ = np.where(stat_ == np.min(stat_))[0][0]
    lambd_old = model.lambdas[idx_]
    beta_old = model.coef()['beta'][idx_]
    intercept_old = model.coef()['intercept'][idx_] if args.use_intercept else 0


    ############################## train ##############################

    for (idx, b) in enumerate(bandwidth):    # different bandwidths

        # ---------------- calculate ga_01, ga_10, a0, a1 -----------------

        ga_01 = np.zeros(n)
        ga_10 = np.zeros(n)
        a0 = np.zeros(n)
        a1 = np.zeros(n)

        ga_01_val = np.zeros(n_v)
        ga_10_val = np.zeros(n_v)
        a0_val = np.zeros(n_v)
        a1_val = np.zeros(n_v)

        continuous_idx = np.delete(np.array(range(p)), discrete_idx)

            
        if args.densityType == "Kernel":
            Z_ = Z[:, continuous_idx]
            Z_val_ = Z_val[:, continuous_idx]

        elif args.densityType == "pcaKernel":
            transformer = PCA(n_components='mle')
            tmp = transformer.fit_transform(np.concatenate((Z[:, continuous_idx], Z_val[:, continuous_idx]), axis=0)) # (n, n_component)
            Z_ = tmp[:n]
            Z_val_ = tmp[n:]
            
        else:
            print("Wrong density estimation method!")
            return
        
        for i in range(n+n_v):

            nume_10 = 0 # numerator
            nume_01 = 0
            deno_10 = 0 # denominator
            deno_01 = 0

            for j in range(n_v):

                if i < n:
                    Zi = Z_[i, :]
                else:
                    Zi = Z_val_[i-n, :]

                Zj = Z_val_[j, :]

                length = Zi.shape[0]
                kb = (np.sqrt(2 * np.pi) * b) ** (- length) * np.exp(- np.linalg.norm(Zi-Zj, ord=2) / (2 * (b ** 2)))

                if len(discrete_idx) > 0:

                    if i < n:
                        Zi = Z[i, discrete_idx]
                    else:
                        Zi = Z_val[i-n, discrete_idx]

                    Zj = Z_val[j, discrete_idx]

                    w_num = (Zi != Zj).sum()
                    kb *= (w ** w_num)

                nume_10 += (kb * Y_val[j] * (1 - Y_star_val[j]))
                nume_01 += (kb * (1 - Y_val[j]) * Y_star_val[j])

                deno_10 += (kb * Y_val[j])
                deno_01 += (kb * (1 - Y_val[j]))
            
            ga_10_ = nume_10 / deno_10
            ga_01_ = nume_01 / deno_01

            if i < n:
                ga_10[i] = ga_10_
                ga_01[i] = ga_01_
                a0[i] = Y_star[i] * ga_01[i] + (1 - Y_star[i]) * (1 - ga_01[i])
                a1[i] = Y_star[i] * (1 - ga_10[i]) + (1 - Y_star[i]) * ga_10[i]

            else:
                ga_10_val[i-n] = ga_10_
                ga_01_val[i-n] = ga_01_
                a0_val[i-n] = Y_star_val[i-n] * ga_01_val[i-n] + (1 - Y_star_val[i-n]) * (1 - ga_01_val[i-n])
                a1_val[i-n] = Y_star_val[i-n] * (1 - ga_10_val[i-n]) + (1 - Y_star_val[i-n]) * ga_10_val[i-n]
        

        args.ga_01 = ga_01
        args.ga_10 = ga_10
        args.a0 = a0
        args.a1 = a1
        args.ga_01_val = ga_01_val
        args.ga_10_val = ga_10_val
        args.a0_val = a0_val
        args.a1_val = a1_val


    
        if np.any(np.isnan(np.concatenate((a0, a1, ga_01, ga_10, a0_val, a1_val, ga_01_val, ga_10_val))) == True):
            continue
            
        # -------------------------- train -----------------------------

        returned_intercepts = np.zeros((N,))

        for t in range(N):

            # ------------------- update intercept ---------------------


            if args.use_intercept:


                res = minimize(fun=glm_fun, x0=intercept_old, args=(beta_old, args))
                intercept = res.x
                returned_intercepts[t] = intercept
                

                if np.abs(intercept - intercept_old) < 1e-4:
                    break

                intercept_old = intercept
                
            # ------------------- update beta ---------------------

            try:


                zero_init = False
                if t == 0 and args.penalty == 'l1':
                    zero_init = True
                # if t == 0 and idx == 0:
                #     zero_init = True
                

                APF_lambda, APF_beta, APF_stat = path_following.APF(
                    args, intercept_old, beta_old, lambd_old, zero_init
                    )

                Criterion_values[idx, t] = APF_stat
                Criterion_betas[idx, t, :] = APF_beta
                Criterion_lambdas[idx, t] = APF_lambda
                
                beta_old = APF_beta
                lambd_old = APF_lambda

            except func_timeout.exceptions.FunctionTimedOut:
                continue

        Returned_intercepts[idx, :] = returned_intercepts
    

    idx = np.argwhere(Criterion_values == np.min(Criterion_values))[0]
    beta_selected = Criterion_betas[idx[0], idx[1], :]
    lambda_selected = Criterion_lambdas[idx[0], idx[1]]
    intercept_selected = Returned_intercepts[idx[0], idx[1]]

    return beta_selected, intercept_selected, lambda_selected




