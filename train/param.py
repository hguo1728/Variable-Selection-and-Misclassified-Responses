import numpy as np
from scipy.optimize import minimize
import pycasso

import func_timeout
from func_timeout import func_set_timeout
from datetime import datetime

from train.utils import info_stat, glm_fun, glm_grad, L1_grad, L1_fun, SCAD_fun, SCAD_grad, MCP_fun, MCP_grad, fast_norm_cdf, fast_norm_pdf
from train import path_following

def param(args):

    n = args.n
    p = args.p
    n_v = args.n_v
    Z = args.Z
    Z_val = args.Z_val
    Y_star = args.Y_star
    Y_val = args.Y_val
    Y_star_val = args.Y_star_val
    

    N = args.N_iter

    Criterion_betas = np.zeros((N, p))
    Criterion_lambdas = np.zeros(N)
    Criterion_values = np.ones(N) * 1e10
    Returned_intercepts = np.zeros(N)
    

    ############################## initialize: pycasso ##############################
    beta_old = np.zeros(p)
    intercept_old = 0
    gamma_old = np.zeros((2 * (p+1), ))
    
    model = pycasso.Solver(np.concatenate((Z, Z_val), axis=0), np.concatenate((Y_star, Y_val), axis=0), lambdas=(100, 0.0015), family='binomial',
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

    ### initialize: ga_01, ga_10, a0, a1

    ga_01_ = sum((1 * (Y_val == 0) + 1 * (Y_star_val == 1)) == 2) / sum(Y_val == 0)
    ga_10_ = sum((1 * (Y_val == 1) + 1 * (Y_star_val == 0)) == 2) / sum(Y_val == 1)

    # main
    ga_01 = np.ones(n) * ga_01_
    ga_10 = np.ones(n) * ga_10_
    a0 = np.zeros(n)
    a1 = np.zeros(n)
    for i in range(n):
        a0[i] = Y_star[i] * ga_01[i] + (1 - Y_star[i]) * (1 - ga_01[i])
        a1[i] = Y_star[i] * (1 - ga_10[i]) + (1 - Y_star[i]) * ga_10[i]

    args.ga_01 = ga_01
    args.ga_10 = ga_10
    args.a0 = a0
    args.a1 = a1
    
    # val
    ga_01_val = np.ones(n_v) * ga_01_
    ga_10_val = np.ones(n_v) * ga_10_
    a0_val = np.zeros(n_v)
    a1_val = np.zeros(n_v)
    for i in range(n_v):
        a0_val[i] = Y_star_val[i] * ga_01_val[i] + (1 - Y_star_val[i]) * (1 - ga_01_val[i])
        a1_val[i] = Y_star_val[i] * (1 - ga_10_val[i]) + (1 - Y_star_val[i]) * ga_10_val[i]
    
    args.ga_01_val = ga_01_val
    args.ga_10_val = ga_10_val
    args.a0_val = a0_val
    args.a1_val = a1_val

    

    ############################## train ##############################

    for t in range(N):    # N rounds -- gamma / intercept and beta

        # -------------------------- update gamma ---------------------------

        # calculate mu
        mu = np.zeros(n)
        for i in range(n):
            tmp = intercept_old + np.dot(beta_old, Z[i, :])
            if args.link_func == "logit":
                mu[i] = np.exp(tmp) / (1 + np.exp(tmp))
            elif args.link_func == "probit":
                mu[i] = fast_norm_cdf(tmp)
            else:
                print("Error: Wrong link function! Return")
                return
        args.mu = mu

        mu_val = np.zeros(n_v)
        for i in range(n_v):
            tmp = intercept_old + np.dot(beta_old, Z_val[i, :])
            if args.link_func == "logit":
                mu_val[i] = np.exp(tmp) / (1 + np.exp(tmp))
            elif args.link_func == "probit":
                mu_val[i] = fast_norm_cdf(tmp)
            else:
                print("Error: Wrong link function! Return")
                return
        args.mu_val = mu_val

            
        gammas_lambdas = np.linspace(lambd_old, max(0.05, lambd_old), 5)
        gammas_num = gammas_lambdas.shape[0]

        a0_ = np.zeros((gammas_num, n))
        a1_ = np.zeros((gammas_num, n))
        ga_01_ = np.zeros((gammas_num, n))
        ga_10_ = np.zeros((gammas_num, n))

        a0_val_ = np.zeros((gammas_num, n_v))
        a1_val_ = np.zeros((gammas_num, n_v))
        ga_01_val_ = np.zeros((gammas_num, n_v))
        ga_10_val_ = np.zeros((gammas_num, n_v))


        gammas = np.zeros((gammas_num, 2 * (p+1)))
        gammas_stat = np.ones(gammas_num) * 1e10
        for k in range(gammas_num):
            res = minimize(fun=fun_lik_gamma, x0=gamma_old, args=(gammas_lambdas[k], args), method='BFGS', jac=grad_Lik)
            gammas[k] = res.x
            gamma_tmp = gammas[k]

            for i in range(n):
                tmp = np.exp(gamma_tmp[0] + np.dot(gamma_tmp[1:p+1], Z[i, :]))
                ga_01_[k, i] = tmp / (1 + tmp)

                tmp = np.exp(gamma_tmp[p+1] + np.dot(gamma_tmp[p+2: 2*(p+1)], Z[i, :]))
                ga_10_[k, i] = tmp / (1 + tmp)

                a0_[k, i] = Y_star[i] * ga_01_[k, i] + (1 - Y_star[i]) * (1 - ga_01_[k, i])
                a1_[k, i] = Y_star[i] * (1 - ga_10_[k, i]) + (1 - Y_star[i]) * ga_10_[k, i]

            for i in range(n_v):
                tmp = np.exp(gamma_tmp[0] + np.dot(gamma_tmp[1:p+1], Z_val[i, :]))
                ga_01_val_[k, i] = tmp / (1 + tmp)

                tmp = np.exp(gamma_tmp[p+1] + np.dot(gamma_tmp[p+2: 2*(p+1)], Z_val[i, :]))
                ga_10_val_[k, i] = tmp / (1 + tmp)

                a0_val_[k, i] = Y_star_val[i] * ga_01_val_[k, i] + (1 - Y_star_val[i]) * (1 - ga_01_val_[k, i])
                a1_val_[k, i] = Y_star_val[i] * (1 - ga_10_val_[k, i]) + (1 - Y_star_val[i]) * ga_10_val_[k, i]
            

            args.a0, args.a1, args.ga_01, args.ga_10 = a0_[k], a1_[k], ga_01_[k], ga_10_[k]
            args.a0_val, args.a1_val, args.ga_01_val, args.ga_10_val = a0_val_[k], a1_val_[k], ga_01_val_[k], ga_10_val_[k]


            if np.any(np.isnan(np.concatenate((args.a0, args.a1, args.ga_01, args.ga_10, args.a0_val, args.a1_val, args.ga_01_val, args.ga_10_val))) == True):
                continue

            gammas_stat[k] = info_stat(args, lambd_old, beta_old, intercept_old)
            
        gamma_idx = np.where(gammas_stat == np.min(gammas_stat))[0][0]
        gamma_old = gammas[gamma_idx, :]

        
        if gammas_stat[gamma_idx] < 1e5:
            args.ga_01 = ga_01_[gamma_idx, :]
            args.ga_10 = ga_10_[gamma_idx, :]
            args.a0 = a0_[gamma_idx, :]
            args.a1 = a1_[gamma_idx, :]

            args.ga_01_val = ga_01_val_[gamma_idx, :]
            args.ga_10_val = ga_10_val_[gamma_idx, :]
            args.a0_val = a0_val_[gamma_idx, :]
            args.a1_val = a1_val_[gamma_idx, :]

        

        # ------------------- update intercept ---------------------

        if args.use_intercept:

            res = minimize(fun=glm_fun, x0=intercept_old, args=(beta_old, args))
            intercept = res.x
            Returned_intercepts[t] = intercept

            if np.abs(intercept - intercept_old) < 1e-4:
                break
            
            intercept_old = intercept
        
        # ------------------------------- update beta --------------------------------

        # train
        try: 

            zero_init = False
            # if t == 0 and args.penalty == 'l1':
            #     zero_init = True

            APF_lambda, APF_beta, APF_stat= path_following.APF(
                args, intercept_old, beta_old, lambd_old, zero_init
                )

            Criterion_values[t] = APF_stat
            Criterion_betas[t, :] = APF_beta
            Criterion_lambdas[t] = APF_lambda
            
            beta_old = APF_beta
            lambd_old = APF_lambda

        except func_timeout.exceptions.FunctionTimedOut:
            print('Time Out! Continue!')
            continue  

    idx = np.where(Criterion_values == np.min(Criterion_values))[0][0]
    beta_selected = Criterion_betas[idx, :]
    lambda_selected = Criterion_lambdas[idx]
    intercept_selected = Returned_intercepts[idx]

    return beta_selected, intercept_selected, lambda_selected

        


#########################################################################################
#                                                                                       #
#                             likelihood: fun and grad -- gamma                         #
#                                                                                       #
#########################################################################################


# ------------------- log-likelihood ------------------

def fun_lik_gamma(gamma, lambd, args):

    '''
        gamma -- gamma[0] (intercept) and gamma[1] (slope)
        mu -- calculated using fixed beta value
    '''

    n = args.n
    p = args.p
    n_v = args.n_v
    Z = args.Z
    Z_val = args.Z_val
    Y_star = args.Y_star
    Y_star_val = args.Y_star_val
    Y_val = args.Y_val
    ga_01 = args.ga_01
    ga_10 = args.ga_10
    mu = args.mu
    mu_val = args.mu_val

    log_lik = 0


    for i in range(n):

        ## 1. Calculate gamma_01 and gamma_10 -- gamma: (2, p+1)
        tmp = np.exp(gamma[0] + np.dot(gamma[1:p+1],  Z[i, :]))
        ga_01 = tmp / (1 + tmp)

        tmp = np.exp(gamma[p+1] + np.dot(gamma[p+2: 2*(p+1)], Z[i, :]))
        ga_10 = tmp / (1 + tmp)
    
        ## 2. Calculate log-likelihood -- non-validation part:
        mu_star = ga_01 + (1 - ga_01 - ga_10) * mu[i]
        log_lik += (Y_star[i] * np.log(mu_star + 1e-6) + (1 - Y_star[i]) * np.log(1 - mu_star + 1e-6))


    for i in range(n_v):

        tmp = np.exp(gamma[0] + np.dot(gamma[1:p+1],  Z_val[i, :]))
        ga_01_val = tmp / (1 + tmp)

        tmp = np.exp(gamma[p+1] + np.dot(gamma[p+2: 2*(p+1)], Z_val[i, :]))
        ga_10_val = tmp / (1 + tmp)

        
        # validation part:

        a0_val = Y_star_val[i] * ga_01_val + (1 - Y_star_val[i]) * (1 - ga_01_val)
        a1_val = Y_star_val[i] * (1 - ga_10_val) + (1 - Y_star_val[i]) * ga_10_val

        log_lik += Y_val[i] * (np.log(mu_val[i] + 1e-6) + np.log(a1_val + 1e-6))\
                    + (1 - Y_val[i]) * (np.log(1 - mu_val[i] + 1e-6) + np.log(a0_val + 1e-6))
        
    
    return (- log_lik / args.num) + SCAD_fun(gamma[1:p+1], lambd) + SCAD_fun(gamma[p+2: 2*(p+1)], lambd)



# ------------------ gradient -------------------

def grad_Lik(gamma, lambd, args):

    n = args.n
    p = args.p
    n_v = args.n_v
    Z = args.Z
    Z_val = args.Z_val
    Y_star = args.Y_star
    Y_star_val = args.Y_star_val
    Y_val = args.Y_val
    ga_01 = args.ga_01
    ga_10 = args.ga_10
    mu = args.mu
    mu_val = args.mu_val

    s = np.zeros((2*(p+1), ))

    for i in range(n):

        ## 1. Calculate gamma_01 and gamma_10

        tmp = np.exp(gamma[0] + np.dot(gamma[1:p+1], Z[i, :]))
        ga_01 = tmp / (1 + tmp)

        tmp = np.exp(gamma[p+1] + np.dot(gamma[p+2: 2*(p+1)], Z[i, :]))
        ga_10 = tmp / (1 + tmp)

        ar = np.concatenate(([1], Z[i, :]))  
        grad_10 = ga_10 * (1 - ga_10) * ar
        grad_01 = ga_01 * (1 - ga_01) * ar
        

        ## 2. Calculate gradient -- non-validation part
        
        if Y_star[i] == 1:
            a0 = ga_01
            a1 = 1 - ga_10
            s1 = (1 - mu[i]) / (a0 * (1 - mu[i]) + a1 * mu[i]) * grad_01 # gradient related to gamma_01
            s2 = - mu[i] / (a0 * (1 - mu[i]) + a1 * mu[i]) * grad_10 # gradient related to gamma_10

            s[: (p+1)] += s1
            s[(p+1):] += s2

        else:
            a0 = 1 - ga_01
            a1 = ga_10
            s1 = - (1 - mu[i]) / (a0 * (1 - mu[i]) + a1 * mu[i]) * grad_01 # gradient related to gamma_01
            s2 = mu[i] / (a0 * (1 - mu[i]) + a1 * mu[i]) * grad_10 # gradient related to gamma_10

            s[: (p+1)] += s1
            s[(p+1): ] += s2

    for i in range(n_v):

        tmp = np.exp(gamma[0] + np.dot(gamma[1:p+1], Z_val[i, :]))
        ga_01 = tmp / (1 + tmp)

        tmp = np.exp(gamma[p+1] + np.dot(gamma[p+2: 2*(p+1)], Z_val[i, :]))
        ga_10 = tmp / (1 + tmp)

        ar = np.concatenate(([1], Z_val[i, :]))  
        grad_10 = ga_10 * (1 - ga_10) * ar
        grad_01 = ga_01 * (1 - ga_01) * ar

        
        if Y_val[i] == 1: # gradient related to gamma_10
            if Y_star[i] == 1:
                tmp = ((-1 / (1 - ga_10 + 1e-6)) * grad_10)
            else:
                tmp = ((1 / (ga_10 + 1e-6)) * grad_10)
            
            s[(p+1):] += tmp
            
        else: # gradient related to gamma_01
            if Y_star[i] == 1:
                tmp = ((1 / ga_01) * grad_01)
            else:
                tmp = ((-1 / (1 - ga_01)) * grad_01)

            s[:(p+1)] += tmp        
    
    return (- s / args.num + np.concatenate(([0], SCAD_grad(gamma[1:p+1], lambd), [0], SCAD_grad(gamma[p+2: 2*(p+1)], lambd)), axis=0))
        

        
