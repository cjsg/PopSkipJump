import time

import numpy as np
from scipy.special import erf, xlogy
from tqdm import tqdm


# Parametric distribution/likelihood family P(y|x, t, s, eps)
def get_py_txse(y, t, x, s, eps):
    """
    :param y : int in {0,1}
    :param t : float in [a,b]
    :param x : float in [0,1]
    :param s : positive float
    :param eps: float in [0., .5]
    """
    sigmoid = lambda x: .5 * np.tanh(2. * x) + .5  # logistic.cdf(4. * x)
    p = eps + (1 - 2 * eps) * sigmoid(s * (x - t))
    return y * p + (1 - y) * (1 - p)


get_py_txse = np.vectorize(get_py_txse)


# Compute E[cos(est_grad, true_grad)]
def get_alpha_(s, theta=0.):  # TODO: include dependence on epsilon
    """
        Computes
            * alpha_ = E[Y(X - theta) X]
                where
                    * p(x-theta) = sigmoid(s(x - theta))
                    * X ~ N(0, 1)  <-- NB: variance of queries is fixed to 1
                    * Y(X) ~ Ber(p(X-theta))

        In words, Y(X-theta) is the binary answer of a classifier Y queried at
        point X, where the classifier's probabilities follow a sigmoid centered
        on theta with (inverse) scale parameter s, and X is sampled according
        to a normal distribution centered on 0 (i.e. at distance theta of the
        sigmoid's center).

        The quantity alpha_ directly yields alpha, which is needed to compute
            * E[cos(est_grad, true_grad)] = 1 / np.sqrt(1 + (d-1) / (n*alpha**2)
    """

    # broadcast s and theta to np.array with same shape
    s = np.array(s)
    theta = np.array(theta)
    b = np.broadcast(s, theta)
    s = np.ones(b.shape) * s
    theta = np.ones(b.shape) * theta

    # Find indeces with infinite scale
    ix_inf = (s == np.inf)
    ix_fin = np.invert(ix_inf)

    # Do computations
    y = np.empty_like(s)
    x_up = (theta[ix_fin] + 1./(2*s[ix_fin])) / np.sqrt(2)
    x_do = (theta[ix_fin] - 1./(2*s[ix_fin])) / np.sqrt(2)
    y[ix_fin] = s[ix_fin] * (erf(x_up) - erf(x_do))
    y[ix_inf] = np.sqrt(2./np.pi) * np.e**(-theta[ix_inf]**2/2.)
    return y


def get_alpha(s, theta, delta, d):
    """
        Computes the same as get_alpha_, but when X follows a Gaussian
        distribution centered on 0 with standard deviation delta / sqrt(d):

            * X ~ N(0, a^2) where a = delta / sqrt(d)

        This amounts to sampling a vector vecX \in R^d from an isotropic
        Gaussian with variance delta^2 = d * a^2 (i.e., vecX ~ N(0, a^2 I_d) )
        and then consider one coordinate of vecX (the coordinate X of vecX
        along the axis of the sigmoid). Clearly, compared to alpha_, this
        amounts to multiplying the inverse scale by delta / sqrt(d) and theta
        by sqrt(d) / delta (because by increasing delta, we 'zoom out', i.e. s
        increases and theta decreases).
    """

    s_ = s * delta / np.sqrt(d)
    theta_ = theta * np.sqrt(d) / delta
    return get_alpha_(s_, theta_)


def get_cos_from_n(n, s=np.infty, theta=0., delta=1., d=10):
    """
        Computes

            * E[cos(est_grad, true_grad)] = 1 / np.sqrt(1 + (d-1) / (n*alpha**2)

        where
            * est_grad = sum_{i=1}^n Y(X-theta) vecX
            * vecX ~ N(0, a^2 I_d)  [see get_alpha]
            * X is the probajection of vecX on the true grad direction
            * theta is the center of the sigmoid along the true grad direction

        In words, get_cos_from_n computes the expected cosinus between the true
        gradient and the gradient estimate that we get by querying a
        probabilistic classifier Y, whose answers follow a sigmoid centered on
        theta with (inverse) scale s, queried at points X ~ N(0, a^2 I_d).

        Note that, we will get the highest expected cos when we center our
        queries X on the center of the classifier's sigmoid theta, which, with
        our conventions here, amounts to taking theta=0.
    """
    alpha = get_alpha(s, theta, delta, d)
    alpha = alpha * np.ones_like(n)  # expand alpha to match dims of n if needed
    ix_nul = (alpha**2) == 0.
    ix_pos = np.invert(ix_nul)

    out = np.empty_like(alpha)
    out[ix_nul] = 0.
    out[ix_pos] = 1. / np.sqrt(1. + (d-1) / (n * alpha**2))
    return out


# Nbr of queries n needed to achieve E[cos(est_grad, true_grad)] = target_cos
def get_n_from_cos(target_cos, s=np.inf, theta=0., delta=1., d=10):
    """
        Computes the number of samples needed to reach a prescribed value
        target_cos of the expected cosine between true and estimated gradient
        E[cos(est_grad, true_grad)], when the estimated gradient is computed as
        in get_cos_from_n, given that we know the parameters (s, theta) of the
        underlying sigmoid, and the Gaussian standard deviation delta / sqrt(d)
        of the queries X (see get_cos_from_n).

        This is the inverse of get_cos_from_n (wrt. to n and target_cos).
    """
    alpha = get_alpha(s, theta, delta, d)
    alpha = alpha * np.ones_like(target_cos)  # expand alpha to match dims of target_cos if needed
    ix_nul = (alpha**2) == 0.
    ix_pos = np.invert(ix_nul)

    out = np.empty_like(alpha)
    out[ix_nul] = np.inf
    out[ix_pos] = (d-1) * target_cos**2 / (alpha[ix_pos]**2 * (1 - target_cos**2))
    return out

def bin_search(
        unperturbed, perturbed, decision_function,
        acquisition_function='I(y,t,s)',  # 'I(y,t,s)', 'I(y,t)', 'I(y,s)', '-E[n]'
        center_on='near_best',  # only used if acq=-E[n]: 'best', 'near_best', 'mean', 'mode', None
        kmax=5000,  # max number of bin search steps
        target_cos=.2,  # targeted E[cos(est_grad, true_grad)]
        delta=.5,  # radius of sphere
        d=1000,  # input dimension
        verbose=False,  # print log info
        window_size=10,
        grid_size=100,
        eps_=.1):
    t_start = time.time()
    Nx, Nt, Ns = grid_size+1, grid_size+1, 31  # Current code assumes Nx = Nt
    Nz = Nx  # candidate center locations = candidate sample location for bin search

    # discretize parameter (search) space
    tt = np.linspace(0, 1, Nt)
    xx = np.linspace(0, 1, Nx)
    yy = [0, 1]
    # ss = np.logspace(-1, 2, Ns)  # s \in [.01, 10.]
    Ns = 1
    ss = np.array([1000])
    eps_ = 1e-6
    eeps = [eps_]  # np.linspace(0., .1, 2)

    ttss = np.stack(np.meshgrid(tt, ss, indexing='ij'))  # 2 x Nt x Ns
    ll = xx[:, np.newaxis] - tt[np.newaxis, :]  # distance matrix: Nx x Nt
    lls, lss = np.meshgrid(ll, ss, indexing='ij')
    lls = lls.reshape(Nx, Nt, Ns)  # Nx x Nt x Ns
    lss = lss.reshape(Nx, Nt, Ns)  # Nx x Nt x Ns
    ii_t = np.arange(Nt)  # indeces of t (useful for later computations)

    def vprint(string):
        if verbose:
            print(string)

    start = time.time()

    # Compute likelihood P(y|t,x)
    Y, T, X, S, E = np.meshgrid(yy, tt, xx, ss, eeps, indexing='ij')
    py_txse = get_py_txse(Y, T, X, S, E)  # [y, t, x, s, eps] axis always in this order
    py_txs = py_txse.sum(axis=4)  # marginalizing out eps
    pt = np.ones((1, Nt, 1, 1)) / Nt  # prior on t
    ps = np.ones((1, 1, 1, Ns)) / Ns  # prior on s
    pts = pt * ps  # prior on (t,s)
    pts_x = pts  # X and (T, S) are independent

    # Compute E[n | (x, t, s)]
    n_tsz = get_n_from_cos(
        s=lss, theta=lls, target_cos=target_cos,
        delta=delta, d=d).transpose([1, 2, 0])
    n_tsz = np.minimum(n_tsz, 1e8)  # for numerical stability
    # if acquisition_function == '-E[n]':
    #     n_ytxsz = n_tsz.reshape(1, Nt, 1, Ns, Nz)

    # Initialize logs
    output = {
        'xxj': [],
        'yyj': [],
        'tts_max': [],
        'tts_map': [],
        'zz_best': [],
        'zz_tmax': [],
        'zz_tmap': [],
        'nn_best_est': [],
        'nn_best_tru': [],
        'nn_tmax_est': [],
        'nn_tmax_tru': [],
        'nn_tmap_est': [],
        'nn_tmap_tru': [],
        # 'n_opt': n_opt,
    }
    tt_preprocessing = time.time() - t_start
    tt_compute_probs, tt_setting_stats, tt_acc_func, tt_max_acquisition, tt_posterior = 0.0, 0.0,0.0,0.0,0.0
    for k in tqdm(range(kmax), desc='bin-search'):
        t_start = time.time()
        # Compute some probabilities / expectations
        pyts_x = py_txs * pts_x
        py_x = pyts_x.sum(axis=(1, 3), keepdims=True)
        pt_x = pts_x.sum(axis=3, keepdims=True)
        n_z = (pts_x.reshape(Nt, Ns, 1) * n_tsz).sum(axis=(0, 1))  # E[n | z]
        tt_compute_probs += (time.time() - t_start)
        t_start = time.time()
        # Compute new stats for logs and stopping criterium
        i_ts_max, j_ts_max = np.unravel_index(pts_x.argmax(), (Nt, Ns))

        ts_max = ttss[:, i_ts_max, j_ts_max]  # Maximum a posteriori (or prior max)
        ts_map = (pts_x.squeeze() * ttss).sum(axis=(1, 2))  # Mean a posteriori (or prior mean)
        iz_best = np.argmin(n_z)
        iz_tmax = pt_x.argmax()
        iz_tmap = int(np.round((pt_x.squeeze() * ii_t).sum()))  # assumes lin-spaced tt
        z_tmax = tt[iz_tmax]
        z_tmap = tt[iz_tmap]
        z_best = tt[iz_best]
        n_zbest_est = n_z[iz_best]
        n_ztmap_est = n_z[iz_tmap]
        n_ztmax_est = n_z[iz_tmax]
        # n_zbest_tru = n_tsz[it_true, is_true, iz_best]  # get_n_from_cos(s_, z_best-t_, target_cos, delta, d)
        # n_ztmax_tru = n_tsz[it_true, is_true, iz_best]  # get_n_from_cos(s_, z_tmax-t_, target_cos, delta, d)
        # n_ztmap_tru = n_tsz[it_true, is_true, iz_tmax]  # get_n_from_cos(s_, z_tmap-t_, target_cos, delta, d)
        tt_setting_stats += time.time() - t_start
        t_start = time.time()
        # Compute acquisition function a(x), x = next sample loc
        if acquisition_function == 'I(y,t,s)':
            # Compute mutual information I(y, (t, s) | (x1,y1), (x2, y2) ... (xj, yj))
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3))
            Hts = -xlogy(pts_x, pts_x).sum(axis=(0, 1, 3))
            Hyts = -xlogy(pyts_x, pyts_x).sum(axis=(0, 1, 3))
            a_x = Hy + Hts - Hyts  # acquisition = mutual info

        elif acquisition_function == 'I(y,t)':
            pyt_x = pyts_x.sum(axis=3, keepdims=True)
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3))
            Ht = -xlogy(pt_x, pt_x).sum(axis=(0, 1, 3))
            Hyt = -xlogy(pyt_x, pyt_x).sum(axis=(0, 1, 3))
            a_x = Hy + Ht - Hyt  # acqui = mutual info

        elif acquisition_function == 'I(y,s)':
            pys_x = pyts_x.sum(axis=1, keepdims=True)
            ps_x = pts_x.sum(axis=1, keepdims=True)
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3))
            Hs = -xlogy(ps_x, ps_x).sum(axis=(0, 1, 3))
            Hys = -xlogy(pys_x, pys_x).sum(axis=(0, 1, 3))
            a_x = Hy + Hs - Hys  # acqui = mutual info

        elif (acquisition_function == '-E[n]' and
              center_on in {'best', 'near_best'}):
            # a(x) = min_z E[n | x, z] = n | y, x, z
            pts_yx = pyts_x / py_x

            if center_on == 'near_best':
                # a(x) = min_{z near E[t|y,x]} E[n | x, z]
                ps_yx = pts_yx.sum(axis=1, keepdims=True)
                pt_yxs = pts_yx / ps_yx
                i_t_yxs = (  # compute index of E[t | y, x, s]
                        ii_t.reshape(1, Nt, 1, 1)  # assumes lin-spaced t
                        * pt_yxs).sum(axis=1, keepdims=True)
                iz_lo = max(int(np.min(i_t_yxs)) - 1, 0)
                iz_up = int(np.round(np.max(
                    (i_t_yxs * ps_yx).sum(axis=3)))) + 1  # E[t | y, x]
            else:
                iz_lo = iz_up = None

            # Compute a(x) = min_z E[n | z, x]
            pts_yxz = pts_yx.reshape(2, Nt, Nx, Ns, 1)
            n_yxz = (pts_yxz * n_ytxsz[..., iz_lo:iz_up]).sum(
                axis=(1, 3), keepdims=True)
            n_yx = np.min(n_yxz, axis=4)
            n_x = (n_yx * py_x).sum(axis=0).squeeze()  # E[n | x]
            a_x = - n_x

        elif (acquisition_function == '-E[n]' and
              center_on in {'mode', 'mean'}):
            # a(x) = E_n[n | zj, x]
            #     'mode': with zj = max-likelihood = argmax_t p(t|y,x)
            #     'mean': with zj = E[t|y,x]

            pts_yx = pyts_x / py_x
            pt_yx = pts_yx.sum(axis=3, keepdims=True)
            if center_on == 'mode':  # z = argmax_t p(t | yj, xj)
                iz_yx = np.argmax(pt_yx, axis=1)  # assumes Nz = Nt
            elif center_on == 'mean':  # z = E[t | y, x]
                iz_yx = np.round((
                                         ii_t.reshape(1, Nt, 1, 1)
                                         * pt_yx).sum(axis=1)).astype(int)
            iz_yx = iz_yx[:, np.newaxis, :, :, np.newaxis]
            n_ytxs = np.take_along_axis(
                n_ytxsz, iz_yx, axis=4)[..., 0]  # drop last dim (z)
            n_yx = (pts_yx * n_ytxs).sum(axis=(1, 3),
                                         keepdims=True)
            n_x = (n_yx * py_x).sum(axis=0).squeeze()  # E[n | x]
            a_x = - n_x

        else:
            raise ValueError
        tt_acc_func += time.time() - t_start
        t_start = time.time()
        # Maximize acquisition function over sampling loc x
        j_amax = np.argmax(a_x)
        xj = xx[j_amax]
        projection = (1 - xj) * unperturbed + xj * perturbed
        yj = int(decision_function(projection[None], freq=1, remember=False))
        # yj = np.random.binomial(n=1, p=get_py_txse(1, 0.2, xj, 10, eps_))
        tt_max_acquisition += time.time() - t_start
        t_start = time.time()
        # Update logs
        # vprint(f'E[n]_lim = {n_opt:.2e}\t E[n] = {n_z[j_amax]:.2e}')
        output['xxj'].append(xj)
        output['yyj'].append(yj)
        output['tts_max'].append(ts_max)
        output['tts_map'].append(ts_map)
        output['zz_best'].append(z_best)
        output['zz_tmax'].append(z_tmax)
        output['zz_tmap'].append(z_tmap)
        output['nn_best_est'].append(n_zbest_est)
        output['nn_tmax_est'].append(n_ztmax_est)
        output['nn_tmap_est'].append(n_ztmap_est)
        if len(output['nn_tmap_est']) > window_size:
            smoothing_kernel = np.ones(10, ) / 10
            exp_n = output['nn_tmap_est'][-window_size:]
            diffs = np.abs(np.diff(exp_n))
            smoothed = int(np.mean(diffs))
            if smoothed == 0:
                break
        # Compute posterior (i.e. new prior) for t
        pyj_txjs = py_txs[yj:(yj + 1), :, j_amax:(j_amax + 1), :]
        pyj_xj = py_x[yj:(yj + 1), :, j_amax:(j_amax + 1), :]
        pts_xyj = pyj_txjs * pts_x / pyj_xj

        # New prior = previous posterior
        pts = pts_xyj
        pts_x = pts  # (t, s) independent of sampling point x
        k += 1
        tt_posterior += time.time() - t_start

    end = time.time()
    # print('Preprocessing: {}'.format(tt_preprocessing))
    # print('Computing Probs: {}'.format(tt_compute_probs))
    # print('Setting Stats: {}'.format(tt_setting_stats))
    # print('Acquisition Func: {}'.format(tt_acc_func))
    # print('Maximizing AccFun: {}'.format(tt_max_acquisition))
    # print('Compute Posterior: {}'.format(tt_posterior))
    # print ('Time to finish: {}'.format(end - start))
    vprint(f'Time to finish: {end - start:.2f} s')

    return output

# output = bin_search(
#     acquisition_function='-E[n]',  # 'I(y,t,s)', 'I(y,t)', 'I(y,s)', '-E[n]'
#     center_on='near_best',  # only used if acq=-E[n]: 'best', 'near_best', 'mean', 'mode', None
#     kmax=1000,  # max number of bin search steps
#     target_cos=.2,  # targeted E[cos(est_grad, true_grad)]
#     delta=.5,  # radius of sphere
#     d=1000,  # input dimension
#     verbose=False,  # print log info
#     eps_=.1)
# assert output['tts_max'][999][0] == 0.2
