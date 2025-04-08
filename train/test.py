import os
import re
import argparse
import numpy as np
from scipy.special import erf

def fast_norm_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def fast_norm_pdf(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

def mu(Z, beta, intercept, type='logit'):
    if type == 'logit':
        val = np.exp(intercept + np.dot(Z, beta))
        mu = val / (1 + val)
    elif type == 'probit':
        tmp = intercept + np.dot(Z, beta)
        mu = fast_norm_cdf(tmp)
    return mu

def fun_test(Z, Y, beta, intercept, type):
    n = Z.shape[0]

    mu_hat = np.zeros(n)
    Y_hat = np.zeros(n)

    for i in range(n): # n
        mu_hat[i] = mu(Z[i], beta, intercept, type)
        Y_hat[i] = 1 if mu_hat[i]>=0.5 else 0
    
    acc = sum(Y == Y_hat) / n
    brier = np.dot(Y - mu_hat, Y - mu_hat) / n

    # calculate auc
    n0 = sum(Y == 0)
    n1 = sum(Y == 1)
    auc = 0
    for i in range(n):
        for j in range(n):
            if Y[i] == 1 and Y[j] == 0 and mu_hat[i] > mu_hat[j]:
                auc += 1
    auc /= (n0 * n1)

    return acc, brier, auc
