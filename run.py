import os
import json

import numpy as np
from train.param import param
from train.semi import semi
from train.test import fun_test

import time
from datetime import datetime
import logging
import argparse
from types import SimpleNamespace



def run(Z, Y_star, Z_val, Y_val, Y_star_val, discrete_idx=[], Z_test=None, Y_test=None, test=False,
        link_func='logit', penalty='scad', use_intercept=True, criterion='gcv',
        model_running='semi', densityType='pcaKernel',
        eta=0.91, R=5, L=0.05, N_iter=5, max_loop=20
        ):
    '''
    -------- Data --------
    * (Z, Y_star): data with noisy response -- numpy array
        dim: (n, p), (n,)
    * (Z_val, Y_val, Y_star_val): validation data with true and noisy response -- numpy array
        dim: (n_v, p), (n_v,), (n_v,)
    * discrete_idx: indices of discrete covariates -- list
    
    -------- Setup --------
    * link_func: link function -- 'logit', 'probit'
    * penalty: penalty function -- 'l1', 'scad', 'mcp'
    * use_intercept: use intercept or not -- True, False
    * criterion: criterion -- 'gcv', 'bic'
    * model_running: training methods -- 'param', 'semi'
    * densityType: density estimation --  'Kernel', 'pcaKernel' (semiparametric method)

    -------- hyper-parameters for optimization --------
    * eta: path following (decreasing coefficient for the sequence of regularization parameters) -- [0.9, 1)
    * R: path following (projection radius) -- positive 
    * L: path following (initial learning rate) -- suﬃciently small positive number
    * N_iter: number of iterations (outer) -- positive integer
    * max_loop: path following: maximum number of loops -- positive integer
    '''

    from types import SimpleNamespace

    args = SimpleNamespace(
        Z=Z,
        Y_star=Y_star,
        Z_val=Z_val,
        Y_val=Y_val,
        Y_star_val=Y_star_val,
        discrete_idx=discrete_idx,
        Z_test=Z_test,
        Y_test=Y_test,
        test=test,
        
        link_func=link_func,
        penalty=penalty,
        use_intercept=use_intercept,
        criterion=criterion,
        model_running=model_running,
        densityType=densityType,

        eta=eta,
        R=R,
        L=L,
        N_iter=N_iter,
        max_loop=max_loop,

        n=Z.shape[0],
        n_v=Z_val.shape[0],
        p=Z.shape[1],
        num=Z.shape[0] + Z_val.shape[0],

        ga_10=None,
        ga_01=None,
        a0=None,
        a1=None,
        mu=None,

        ga_10_val=None,
        ga_01_val=None,
        a0_val=None,
        a1_val=None,
        mu_val=None
    )


    # parser = argparse.ArgumentParser()
    # args=parser.parse_args()

    # args.Z = Z
    # args.Y_star = Y_star
    # args.Z_val = Z_val 
    # args.Y_val = Y_val
    # args.Y_star_val = Y_star_val
    # args.discrete_idx = discrete_idx
    # args.Z_test = Z_test
    # args.Y_test = Y_test
    # args.test = test
    

    # args.link_func = link_func
    # args.penalty = penalty
    # args.use_intercept = use_intercept
    # args.criterion = criterion
    # args.model_running = model_running
    # args.densityType = densityType

    # args.eta = eta
    # args.R = R
    # args.L = L
    # args.N_iter = N_iter
    # args.max_loop = max_loop

    # args.n = Z.shape[0]
    # args.n_v = Z_val.shape[0]
    # args.p = Z.shape[1]
    # args.num = args.n + args.n_v
    # args.ga_10 = None
    # args.ga_01 = None
    # args.a0 = None
    # args.a1 = None
    # args.mu = None
    # args.mu_val = None
    # args.ga_10_val = None
    # args.ga_01_val = None
    # args.a0_val = None
    # args.a1_val = None
    # args.mu_val = None

    if model_running == 'param':
        beta_, intercept_, lambd_ = param(args)
    elif model_running == 'semi':
        beta_, intercept_, lambd_ = semi(args)
    else:
        print("Wrong model_running choice! Return")
        return
    
    result = {}
    result["beta"] = beta_
    result["intercept"] = intercept_
    result["lambd"] = lambd_

    if test and Z_test is not None and Y_test is not None:
        acc, brier, auc = fun_test(Z_test, Y_test, beta_, intercept_, args.link_func)
        result["acc"] = acc
        result["brier"] = brier
        result["auc"] = auc
    
    return result









    