import numpy as np
from scipy.optimize import minimize
from scipy.special import erf


import func_timeout
from func_timeout import func_set_timeout



########################################################################
#        
#                  Calculate GCV and BIC statistic
#
########################################################################


def info_stat(args, lambd, beta, intercept=0):
    '''
        input:
            Z -- covariates (free of error)
            Y_input -- true label (validation part) & noisy label (non-validation part)
            n_v -- validation size
            beta -- beta
            ga -- ga_01 = ga_10 = ga
            lambd -- lambda
            penalty -- 'l1' or 'scad' or 'mcp'
            criterion -- 'gcv' or 'bic'
        output:
            beta_GCV
            beta_BIC
    '''

    n = args.n
    p = args.p
    n_v = args.n_v
    Z = args.Z
    Z_val = args.Z_val
    Y_star = args.Y_star
    Y_val = args.Y_val
    ga_01 = args.ga_01
    ga_10 = args.ga_10

    idx = np.nonzero(beta)[0]
    s = np.shape(idx)[0]
    
    # V: the covariates included in the model
    V = Z[:, idx]
    V_val = Z_val[:, idx]

    # ------------- Sigma_lambda -----------
    # Sigma_lambda: diagonal matrix with elements equal to p'_{lambda}(|beta_j|)/|beta_j| for non-vanishing beta_j
    sigmas = np.zeros(s)
    
    for k in range(s):
        if args.penalty == 'l1':
            sigmas[k] = lambd / (np.abs(beta[idx[k]]) + 1e-6)
        elif args.penalty == 'scad':
            if np.abs(beta[idx[k]]) <= lambd:
                sigmas[k] = lambd / (np.abs(beta[idx[k]])+ 1e-6)
            elif lambd < np.abs(beta[idx[k]]) <= 3.7 * lambd:
                sigmas[k] = (3.7 * lambd - np.abs(beta[idx[k]])) / ((3.7 - 1) * np.abs(beta[idx[k]]) + 1e-6)
            else:
                sigmas[k] = 0
        elif args.penalty == 'mcp':
            if np.abs(beta[idx[k]]) <= 3.7 * lambd:
                sigmas[k] = (lambd - np.abs(beta[idx[k]]) / 3.7) / (np.abs(beta[idx[k]]) + 1e-6)
            else:
                sigmas[k] = 0
        else:
            print('Error: invalid penalty!')
            return
    
    Sigma = np.diag(sigmas)

    # --------- I: Fisher Matrix ----------
    I = np.zeros((s, s))

    # main data
    mu = np.zeros(n) # mu_hat (mu_star_hat)
    for i in range(n):
        tmp = intercept + np.dot(Z[i, :], beta)
        if args.link_func == 'logit':
            mu[i] = np.exp(tmp) / (1 + np.exp(tmp))
            grad =  mu[i] * (1 - mu[i]) * V[i, :] # d(mu) / d(beta)
        elif args.link_func == 'probit':
            mu[i] = fast_norm_cdf(tmp)
            grad = fast_norm_pdf(tmp) * V[i, :] # d(mu) / d(beta)
        else:
            print("Error: Wrong link function! Return")
            return

        # initialize: no ga_01/ga_10/a1/a0
        if args.ga_01 is None or args.ga_10 is None or args.a1 is None or args.a0 is None:
            I +=  np.matmul(grad.reshape(s, 1), grad.reshape(1, s)) / (mu[i] * (1 - mu[i]) + 1e-6)
        
        # after training:
        else:
            q_i = (1 - ga_01[i] - ga_10[i]) ** 2 / (mu[i] * (1 - mu[i]) + 1e-6)
            I += q_i * np.matmul(grad.reshape(s, 1), grad.reshape(1, s)) 
            # mu_star
            mu[i] = ga_01[i] + (1 - ga_01[i] - ga_10[i]) * mu[i]
    
    # validation data
    mu_val = np.zeros(n_v) # mu_hat 
    for i in range(n_v):
        tmp = intercept + np.dot(Z_val[i, :], beta)
        if args.link_func == 'logit':
            mu_val[i] = np.exp(tmp) / (1 + np.exp(tmp))
            grad =  mu_val[i] * (1 - mu_val[i]) * V_val[i, :] # d(mu) / d(beta)
        elif args.link_func == 'probit':
            mu_val[i] = fast_norm_cdf(tmp)
            grad = fast_norm_pdf(tmp) * V_val[i, :] # d(mu) / d(beta)
        else:
            print("Error: Wrong link function! Return")
            return

        # validation part
        q_i = 1 / (mu_val[i] * (1 - mu_val[i]) + 1e-6)
        I += q_i * np.matmul(grad.reshape(s, 1), grad.reshape(1, s))

    I /= (n + n_v)

    # -------- df: degrees of freedom ----------
    # print(np.diag(I + Sigma))
    df = np.trace(np.dot(I, np.linalg.inv(I + Sigma)))

    # -------- D: deviance of the model -----------
    D = 0

    for i in range(n):
        if Y_star[i] == 0:
            val = np.log((1 - Y_star[i]) / (1 - mu[i] + 1e-6))
        else:
            val = np.log(Y_star[i] / (mu[i] + 1e-6))
        D += (2 * val)
    
    for i in range(n_v):
        if Y_val[i] == 0:
            val = np.log((1 - Y_val[i]) / (1 - mu_val[i] + 1e-6))
        else:
            val = np.log(Y_val[i] / (mu_val[i] + 1e-6))
        D += (2 * val)
    
    N = n + n_v
    
    if args.criterion == 'gcv':
        GCV = D / (N * ((1 - df / N) ** 2))
        return GCV
    elif args.criterion == 'bic':
        BIC = D + 2 * np.log(N) * df
        return BIC
    else:
        print("Wrong Criterion!")
        return




#########################################################################
# 
#             likelihood: (negative) log-likelihood and gradient    
#       
#########################################################################
    

# ------------------- logit link:  ------------------

def glm_fun(intercept, beta, args):

    n = args.n
    p = args.p
    n_v = args.n_v
    Z = args.Z
    Z_val = args.Z_val
    Y_star = args.Y_star
    Y_val = args.Y_val
    ga_01 = args.ga_01
    ga_10 = args.ga_10
    a0_val = args.a0_val
    a1_val = args.a1_val

    log_lik = 0

    for i in range(n):

        tmp = intercept + np.dot(beta, Z[i, :])
        if args.link_func == 'logit':
            mu = np.exp(tmp) / (1 + np.exp(tmp))
        elif args.link_func == 'probit':
            mu = fast_norm_cdf(tmp)
        else:
            print("Error: Wrong link function! Return")
            return

        # non-validation part:
        mu_star = ga_01[i] + (1 - ga_01[i] - ga_10[i]) * mu

        tmp = Y_star[i] * np.log(mu_star + 1e-6) + (1 - Y_star[i]) * np.log(1 - mu_star + 1e-6)
        log_lik += tmp
    
    for i in range(n_v):
        tmp = intercept + np.dot(beta, Z_val[i, :])
        if args.link_func == 'logit':
            mu = np.exp(tmp) / (1 + np.exp(tmp))
        elif args.link_func == 'probit':
            mu = fast_norm_cdf(tmp)
        else:
            print("Error: Wrong link function! Return")
            return
        
        # validation part:
        tmp = Y_val[i] * (np.log(mu + 1e-6) + np.log(a1_val[i] + 1e-6))\
                    + (1 - Y_val[i]) * (np.log(1 - mu + 1e-6) + np.log(a0_val[i] + 1e-6))
        log_lik += tmp
    
    N = n + n_v

    return (- log_lik / N)


    
def glm_grad(intercept, beta, args):

    n = args.n
    p = args.p
    n_v = args.n_v
    s = np.zeros(p)

    a1 = args.a1
    a0 = args.a0
    Z = args.Z
    Z_val = args.Z_val
    Y_val = args.Y_val

    for i in range(n):
        tmp = intercept + np.dot(beta, Z[i, :])
        if args.link_func == 'logit':
            mu = np.exp(tmp) / (1 + np.exp(tmp))
            grad = mu * (1 - mu) * Z[i, :] # d(mu) / d(beta)
        elif args.link_func == 'probit':
            mu = fast_norm_cdf(tmp)
            grad = fast_norm_pdf(tmp) * Z[i, :] # d(mu) / d(beta)
        else:
            print("Error: Wrong link function! Return")
            return
        s += ((a1[i] - a0[i]) / (a1[i] * mu + a0[i] * (1 - mu) + 1e-6)) * grad
    
    for i in range(n_v):
        tmp = intercept + np.dot(beta, Z_val[i, :])
        if args.link_func == 'logit':
            mu = np.exp(tmp) / (1 + np.exp(tmp))
            grad = mu * (1 - mu) * Z_val[i, :] # d(mu) / d(beta)
        elif args.link_func == 'probit':
            mu = fast_norm_cdf(tmp)
            grad = fast_norm_pdf(tmp) * Z_val[i, :] # d(mu) / d(beta)
        else:
            print("Error: Wrong link function! Return")
            return
        s += ((Y_val[i] - mu) / (mu * (1 - mu) + 1e-6)) * grad

    N = n + n_v  

    return (- s / N)




#########################################################################
#                                                                       #
#                        penalty: fun and grad                          #
#                                                                       #
#########################################################################


# -------------------- L1 --------------------

def L1_fun(beta, lambd):
    return lambd * np.linalg.norm(beta, ord=1)

def L1_grad(beta, lambd):
    return lambd * np.sign(beta)


# -------------------- SCAD --------------------

def SCAD_fun(beta, lambd, a=3.7):
    p = beta.shape[0]
    fun = 0

    for i in range(p):
        u = np.abs(beta[i])

        if u < lambd:
            tmp = lambd * u
        elif u < a * lambd:
            tmp = (a * lambd * u - (u ** 2) / 2 - (lambd ** 2) / 2) / (a - 1)
        else:
            tmp = (lambd ** 2) * (a + 1) / 2
        
        fun += tmp

    return fun
    
def SCAD_grad(beta, lambd, a=3.7):
    p = beta.shape[0]
    grad = np.zeros(p)

    for i in range(p):
        u = np.abs(beta[i])

        if u < lambd:
            tmp = lambd
        elif u < a * lambd:
            tmp = (a * lambd - u) / (a - 1)
        else:
            tmp = 0
        
        grad[i] = tmp * np.sign(beta[i])

    return grad


# -------------------- MCP --------------------

def MCP_fun(beta, lambd, a=3.7):
    p = beta.shape[0]
    fun = 0

    for i in range(p):
        u = np.abs(beta[i])

        if u < a * lambd:
            tmp = lambd * u - (u ** 2) / (2 * a)
        else:
            tmp = a * (lambd ** 2) / 2
        
        fun += tmp

    return fun


def MCP_grad(beta, lambd, a=3.7):
    p = beta.shape[0]
    grad = np.zeros(p)

    for i in range(p):
        u = np.abs(beta[i])

        if u < a * lambd:
            tmp = (lambd - u / a) 
        else:
            tmp = 0
        
        grad[i] = tmp * np.sign(beta[i])

    return grad



#########################################################################
#                                                                       #
#                        probit link: cdf $ pdf                         #
#                                                                       #
#########################################################################



def fast_norm_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def fast_norm_pdf(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    



