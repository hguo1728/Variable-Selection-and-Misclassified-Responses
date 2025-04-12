
######################################################################
#                                                                    #
#      Wang, Liu, and Zhang, 2014, Approximate Path Following        #
#                                                                    #
######################################################################


import numpy as np
from train.utils import glm_fun, glm_grad, info_stat, L1_fun, L1_grad, SCAD_fun, SCAD_grad, MCP_fun, MCP_grad


def APF(args, intercept, beta_init=None, lambd_init=None, zero_init=False):
    '''
    approxaimate regularization path-following method

    input:
            Z, Y_star, Y, Y_input, Y_vali -- data
            intercept -- intercept
            ga_01, ga_10, a0, a1 -- nonparametric part
    output:
            w(beta), 
    '''

    # time1 = time.time()

    n = args.n
    p = args.p
    n_v = args.n_v

    # ---------------- set parameters ----------------
    
    eps_opt = 1e-5
    L_min = 1e-6

    if args.L is None:
        L = L_min
    else:
        L = args.L

    # radius: projection onto the l2-ball
    if args.R is None:
        R = 2 * np.linalg.norm(beta_init, ord=2)
    else:
        R = args.R
    
    
    # ---------------- calculate lambdas -----------------

    if zero_init:
        lambd_init = None
        beta_init = None

    if lambd_init is None:
        tmp = glm_grad(intercept, np.zeros(p), args)  # lambda_0 = \|grad(L(0))\|_infinity
    else:
        tmp = lambd_init

    # tmp = np.maximum(tmp, 0.01)

    lambda_max = np.max(np.abs(tmp))    # lambda_0
    lambda_tgt = np.sqrt(np.log(p) / args.num) / 5  # lambda_N: proportional to sqrt(log(p) / n)

    N1 = int(np.floor(np.log(lambda_max / lambda_tgt + 1e-6) / np.log(1 / args.eta)))
    N_max = args.max_loop

    round_check = 8

    

    # ---------------- outer loop: different lambdas --------------

    i = 0

    lambd = lambda_max
    betas = np.zeros((N_max, p))
    stat_values = np.ones(N_max) * 1e10
    stat_lambdas = np.zeros(N_max)

    lambd_return = 0
    beta_return = np.zeros(p)
    stat_return = 0

    beta_old = np.zeros(p)
    if beta_init is not None: 
        beta_old = beta_init


    while i < N_max:

        stat_lambdas[i] = lambd
        eps = lambd / 4

        if i > 2: 
            cri_1 = np.linalg.norm(betas[i-1, :] - betas[i-2, :], ord=2) < 1e-4
            cri_2 = stat_values[i-1] > np.maximum(stat_values[i-2], stat_values[i-3])

            if cri_1 or (i >= round_check and cri_2):

                lambd = lambd * 4
                stat_lambdas[i] = lambd
                eps = eps_opt

                L, betas[i, :] = ProximalGradient(args, intercept, N_max, lambd, eps, beta_old,  L, L_min, R)
                stat_values[i] = info_stat(args, lambd, betas[i, :], intercept)

                i += 1

                break
        
        L, betas[i, :] = ProximalGradient(args, intercept, N_max, lambd, eps, beta_old,  L, L_min, R)
        beta_old = betas[i, :].copy()
        stat_values[i] = info_stat(args, lambd, beta_old, intercept)

        lambd *= args.eta
        i += 1

    stat_idx = np.where(stat_values == np.min(stat_values))[0][0]
    lambd_return = stat_lambdas[stat_idx]
    stat_return = stat_values[stat_idx]
    beta_return = betas[stat_idx, :]

    return lambd_return, beta_return, stat_return

        

#########################################################################
# 
#                  proximal descent and line search
#       
#########################################################################

# ------------------- ProximalGradient -------------------------

def ProximalGradient(args, intercept, max_loop, lambd, eps, beta0,  L0, L_min, R):
    
    '''
    ProximalGradient(lambd, eps, beta0,  L0, R) -> beta, L (iteration t)
    input: 
        Z, Y_star, Y_vali, ga_01, ga_10, a0, a1, intercept, reg -- detremine the objective function + penalty
        max_loop -- maximum number of loops
        lambd -- lambda(t) = eta^t * lambda_0 
        beta0 -- beta(t): initial beta 
        L0 -- L(t): initial learning rate 
        L_min -- L_min = 10^-6 (hyper-parameter)
        R -- projection radius (hyper-parameter)
    output:
        L -- L(t)
        beta -- beta(t)
    '''

    L = L0 
    beta = beta0
    loop = 0

    while(loop < max_loop):
        loop += 1
        L = np.maximum(L_min, L / 2)

        ############### Line search -- Line 8 of Algorihtm 3 (Algo. 2) ###############
        L, beta = LineSearch(args, intercept, max_loop, lambd, beta.copy(), L, R)


        ############### check stopping criterion -- Line 9 of Algorihtm 3 ###############

        # ----- Eq. (3.16) -----

        # gradient of L(beta)
        grad1 = glm_grad(intercept, beta, args)

        # gradient of Q(beta) = penalty - L1
        if args.penalty == 'l1':
            grad2 = 0
        elif args.penalty == 'scad':
            grad2 = SCAD_grad(beta, lambd) - L1_grad(beta, lambd)
        elif args.penalty == 'mcp':
            grad2 = MCP_grad(beta, lambd) - L1_grad(beta, lambd)
        grad = grad1 + grad2


        # grad(tilde{L}(beta)) + lambd * subgradient
        chec_cri = grad + lambd * np.sign(beta)
        idx = np.where(beta == 0)[0]
        for j in idx:
            if grad[j] >= 0: chec_cri[j] += lambd
            else: chec_cri[j] -= lambd
            
        # omega
        omega = np.max(np.abs(chec_cri))

        # check:
        if omega <= eps:
            break

    return L, beta
    

# ------------------- LineSearch -------------------------

def LineSearch(args, intercept, max_loop, lambd, beta0, L0, R):
    
    '''
    LineSearch(lambd, beta0, L0, R) -> beta, L (iteration (t, k))
    search for the best L and compute the corresponding beta
    input: 
        Z, Y_star, Y_vali, ga_01, ga_10, a0, a1, intercept, reg -- detremine the objective function + penalty
        max_loop -- maximum number of loops
        lambd -- lambda(t) = eta^t * lambda_0 
                (from outer loop t)
        beta0 -- beta(t, k-1): initial beta 
                (from previous inner loop (t, k-1): proximal-gradient) 
        L0 -- max{L_min, L(t, k-1)/2}: initial learning rate 
                (from previous inner loop (t, k-1): proximal-gradient)
        R -- projection radius
    output:
        L -- L(t, k)
        beta -- beta(t, k)
    '''

    n = args.n
    p = args.p
    n_v = args.n_v

    # initialization

    L = L0 / 2
    beta_old = beta0
    fun_old = glm_fun(intercept, beta_old, args)
    if args.penalty == 'scad':
        tmp = SCAD_fun(beta_old, lambd) - L1_fun(beta_old, lambd)
        fun_old += tmp
    elif args.penalty == 'mcp':
        tmp = MCP_fun(beta_old, lambd) - L1_fun(beta_old, lambd)
        fun_old += tmp
    loop = 0

    while(loop < max_loop):
        loop += 1
            
        ############### Update scheme: Eq. (3.9) -- Line 5 of Algorithm 2 ###############

        # ----- soft-thresholding operator -----

        # Eq. (3.11)

        # gradient of L(beta)
        grad1 = glm_grad(intercept, beta_old, args)

        # gradient of Q(beta) = penalty - L1
        if args.penalty == 'l1':
            grad2 = np.zeros(p)
        elif args.penalty == 'scad':
            grad2 = SCAD_grad(beta_old, lambd) - L1_grad(beta_old, lambd)
        elif args.penalty == 'mcp':
            grad2 = MCP_grad(beta_old, lambd) - L1_grad(beta_old, lambd)
        grad = grad1 + grad2

        beta_bar = beta_old - grad / L 


        # Eq. (3.10)

        beta = np.zeros(p)
        idx = np.where(np.abs(beta_bar) > lambd / L)[0]
        
        beta[idx] = np.sign(beta_bar[idx]) * (np.abs(beta_bar[idx]) - lambd / L)

        # ----- Project on the l_2 ball: Eq. (3.12) ----- 

        radius = np.linalg.norm(beta, ord=2)
        if radius > R:
            beta *= (R / radius)
            
        
        ############### check stopping criterion -- Line 6,7 of Algorithm 2 ###############

        # (1) objective function: phi(beta) = L(beta) + penalty(beta)

        # L(beta)
        fun1 = glm_fun(intercept, beta, args)

        # penalty(beta)
        if args.penalty == 'l1':
            fun2 = L1_fun(beta, lambd)
        elif args.penalty == 'scad':
            fun2 = SCAD_fun(beta, lambd)
        elif args.penalty == 'mcp':
            fun2 = MCP_fun(beta, lambd)
        fun = fun1 + fun2

        # (2) local quadratic approximation: psi(beta; beta_old) -- Eq. (3.7)
        tmp = beta - beta_old
        quad = fun_old + np.dot(grad, tmp) + 0.5 * L * np.dot(tmp, tmp) + L1_fun(beta, lambd)

        # check:
        if quad >= fun:
            break

        L *= 2
        beta_old = beta
        fun_old = fun - L1_fun(beta, lambd) # = phi - L1 penalty

    return L, beta




