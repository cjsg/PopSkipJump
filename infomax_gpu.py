import time
from math import pi, sqrt, log10
import torch
from scipy.special import erf  # , xlogy
from tqdm import tqdm
# from tqdm.notebook import tqdm  # tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Parametric distribution/likelihood family P(y|x, t, s, eps)
def get_py_txse(y, t, x, s, eps):
    """
    :param y : int in {0,1}
    :param t : float in [a,b]
    :param x : float in [0,1]
    :param s : positive float
    :param eps: float in [0., .5]
    """
    sigmoid = lambda x: torch.where(torch.isinf(x),  # logistic.cdf(4. * x)
                                    torch.sign(x),
                                    .5 * torch.tanh(2. * x) + .5)
    if type(x) != torch.Tensor:
        x = torch.tensor(x)

    z = torch.zeros((), device=x.device)
    prod = torch.where(x == t, z, s * (x - t))
    p = eps + (1 - 2 * eps) * sigmoid(prod)
    return y * p + (1 - y) * (1 - p)


# get_py_txse = np.vectorize(get_py_txse) # not used, right?

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
            * E[cos(est_grad, true_grad)] = 1 / sqrt(1 + (d-1) / (n*alpha**2)
    """

    # broadcast s and theta to torch.array with same shape
    if type(s) != torch.Tensor:
        s = torch.tensor(s)
    if type(theta) != torch.Tensor:
        theta = torch.tensor(theta)
    s, theta = torch.broadcast_tensors(s, theta)

    # Find indeces with infinite scale
    ix_inf = (s == float("Inf"))
    ix_fin = ~ix_inf

    # Do computations
    y = torch.empty_like(s)
    x_up = (theta[ix_fin] + 1. / (2 * s[ix_fin])) / sqrt(2)
    x_do = (theta[ix_fin] - 1. / (2 * s[ix_fin])) / sqrt(2)
    y[ix_fin] = s[ix_fin] * (torch.erf(x_up) - torch.erf(x_do))
    y[ix_inf] = sqrt(2. / pi) * torch.exp(-theta[ix_inf] ** 2 / 2.)
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
    if type(d) != torch.Tensor:
        d = torch.tensor(d, dtype=torch.float32)
    d.type(torch.float32)

    s_ = s * delta / torch.sqrt(d)
    theta_ = theta * torch.sqrt(d) / delta
    return get_alpha_(s_, theta_)


def get_cos_from_n(n, s=float('Inf'), theta=0., delta=1., d=10):
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
    if type(n) != torch.Tensor:
        n = torch.tensor(n, dtype=torch.float32)

    alpha = get_alpha(s, theta, delta, d)
    n = n.to(alpha.device)
    alpha, n = torch.broadcast_tensors(alpha, n)
    ix_nul = (alpha ** 2) == 0.
    ix_pos = ~ix_nul

    out = torch.empty_like(alpha)
    out[ix_nul] = 0.
    out[ix_pos] = 1. / torch.sqrt(1. + (d-1) / (n[ix_pos] * alpha[ix_pos] ** 2))
    return out


# Nbr of queries n needed to achieve E[cos(est_grad, true_grad)] = target_cos
def get_n_from_cos(target_cos, s=float('Inf'), theta=0., delta=1., d=10):
    """
        Computes the number of samples needed to reach a prescribed value
        target_cos of the expected cosine between true and estimated gradient
        E[cos(est_grad, true_grad)], when the estimated gradient is computed as
        in get_cos_from_n, given that we know the parameters (s, theta) of the
        underlying sigmoid, and the Gaussian standard deviation delta / sqrt(d)
        of the queries X (see get_cos_from_n).

        This is the inverse of get_cos_from_n (wrt. to n and target_cos).
    """
    if type(target_cos) != torch.Tensor:
        target_cos = torch.tensor(target_cos, dtype=torch.float32)

    alpha = get_alpha(s, theta, delta, d)  # returns a tensor
    target_cos = target_cos.to(alpha.device)
    alpha, target_cos = torch.broadcast_tensors(alpha, target_cos)
    ix_nul = (alpha ** 2) == 0.
    ix_pos = ~ix_nul

    out = torch.empty_like(alpha)
    out[ix_nul] = float('Inf')
    out[ix_pos] = (d - 1) * target_cos[ix_pos] ** 2 / (
            alpha[ix_pos] ** 2 * (1 - target_cos[ix_pos] ** 2))
    return out


### Utilities ###

def unravel_index(index, shape):
    '''Mimics np.unravel_index'''
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


# TODO: check that this works as well as scipy for large values
def xlogy(x, y):
    if type(x) != torch.Tensor:
        x = torch.tensor(x)
    if type(y) != torch.Tensor:
        y = torch.tensor(y)

    z = torch.zeros((), device=x.device)
    return x * torch.where(x == 0., z, torch.log(y))


def plot_acquisition(k, xx, a_x, pts_x, ttss, output, acq_func):
    f, axs = plt.subplots(1, 2, figsize=(15, 5))

    xx = xx.cpu()
    a_x = a_x.cpu()
    pts_x = pts_x.cpu()
    ttss = ttss.cpu()

    axs[0].plot(xx, a_x)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel(acq_func)

    y_do, y_up = axs[0].get_ylim()
    xxj, yyj = output['xxj'], output['yyj']
    axs[0].scatter(xxj[-200:], (y_do + torch.tensor(yyj) * (y_up - y_do))[-200:],
                   color='C1', marker='x', alpha=.5)

    vmin = max(pts_x.min(), 1e-7)  # alternatively, fix 1e-7
    vmax = pts_x.max()  # alternatively, fix 1.
    fig1 = axs[1].pcolor(ttss[0], ttss[1], pts_x[0, :, 0, :],
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    axs[1].set_yscale('log')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('s')
    f.colorbar(fig1, ax=axs[1])

    ts_map = output['tts_map'][-1]
    ts_max = output['tts_max'][-1]
    n_zbest = output['nn_best_est'][-1]
    f.suptitle('k=%d    En=%.1e    '
               'map=(%4.2f, %4.2f)   '
               'max=(%4.2f, %4.2f)' % (
                   k, n_zbest, ts_map[0],
                   ts_map[1], ts_max[0], ts_max[1]))
    plt.show()


def get_bernoulli_probs(xx, unperturbed, perturbed, model_interface, true_label):
    dims = [-1] + [1] * unperturbed.ndim
    xx = xx.view(dims)
    batch = (1 - xx) * unperturbed + xx * perturbed
    probs = model_interface.get_probs_(batch)
    probs = probs[:, true_label]
    return probs


def bin_search(
        unperturbed, perturbed, model_interface,
        acq_func='I(y,t,s)', center_on='near_best', kmax=5000, target_cos=.2,
        delta=.5, d=1000, verbose=False, window_size=10, grid_size=100,
        eps_=.1, device=None, true_label=None, plot=False, prev_t=None,
        prev_s=None, prior_frac=1., queries=5):
    '''
        acq_func    (str)   Must be one of
                            ['I(y,t,s)', 'I(y,t)', 'I(y,s)', '-E[n]']
        center_on   (str)   Only used if acq=-E[n] Must be one of
                            ['best', 'near_best', 'mean', 'mode', None]
        kmax:       (int)   max number of bin search steps
        target_cos  (float) targeted E[cos(est_grad, true_grad)]
        delta       (float) radius of sphere
        d           (int)   input dimension
        verbose     (bool)  print log info
        window_size (int)   size of smoothing window for stopping criterium
        grid_size   (int)   grid size used for discretization of t
        eps_        (float) noise level used
        device      (str)   which device to use ('cuda' or 'cpu')
        plot        (bool)  to plot or not to plot
        prev_t      (float) previous estimate of the sigmoid center t (None)
        prev_s      (float) previous estimate of the sigmoid inverse-scale s (None)
        prior_frac  (float) how much to reduce the a priori search interval
                            to the left and to the right of prev_t and prev_s
        queries (int) how many queries to perform in each iteration

        Notation conventions in the code:
            * t     center of sigmoid
            * s     inverse scale of sigmoid
            * eps   noise levels at +-infty
            * z     centering point for gradient sampling
            * pt_x, pts_x, ...
                    p(t|x), p(t,s|x), ...
            * n_z, n_tsz
                    E[n|z], E[n|t,s,z]
    '''

    t_start = time.time()

    if prev_t is None:
        t_lo, t_hi = 0., 1.
        Nt = grid_size + 1
    else:
        t_lo = max(prev_t - prior_frac, 0.)
        t_hi = min(prev_t + prior_frac, 1.)
        Nt = int(grid_size * 2 * prior_frac) + 1

    Nx = grid_size + 1  # number sampling locations
    Nz = Nt  # possible sigmoid centers = possible centers of sampling ball

    if prev_s is None:
        s_lo, s_hi = -1., 2.
        Ns = 31
    else:
        s_lo = log10(prev_s) - prior_frac * 3
        s_hi = log10(prev_s) + prior_frac * 3
        Ns = int(prior_frac * 30) + 1


    # discretize parameter (search) space
    dtype = torch.float32
    tt = torch.linspace(t_lo, t_hi, Nt, dtype=dtype, device=device)
    zz = torch.linspace(t_lo, t_hi, Nz, dtype=dtype, device=device)  # center of sampling ball
    xx = torch.linspace(0., 1., Nx, dtype=dtype, device=device)
    yy = torch.tensor([0, 1], dtype=dtype, device=device)
    ss = torch.logspace(s_lo, s_hi, Ns, dtype=dtype, device=device)  # s \in [.01, 10.]
    # ss[-1] = float("Inf")   # xlogy may not work when s is infinite
    eeps = torch.tensor([eps_], dtype=dtype, device=device)  # torch.linspace(0., .1, 2)

    ttss = torch.stack(torch.meshgrid(tt, ss))  # 2 x Nt x Ns  (numpy indexing='ij')
    ll = zz[:, None] - tt[None, :]  # distance matrix: Nz x Nt
    lls, lss = torch.meshgrid(ll.flatten(), ss)
    lls = lls.reshape(Nz, Nt, Ns)  # Nx x Nt x Ns
    lss = lss.reshape(Nz, Nt, Ns)  # Nx x Nt x Ns
    ii_t = torch.arange(Nt, device=device)  # indeces of t (useful for later computations)

    pp = get_bernoulli_probs(xx, unperturbed, perturbed, model_interface, true_label)

    def vprint(string):
        if verbose:
            print(string)

    start = time.time()

    # Compute likelihood P(y|t,x)
    Y, T, X, S, E = torch.meshgrid(yy, tt, xx, ss, eeps)
    py_txse = get_py_txse(Y, T, X, S, E)  # [y, t, x, s, eps] axis always in this order
    py_txs = py_txse.sum(axis=4)  # marginalizing out eps
    pt = torch.ones((1, Nt, 1, 1), device=device) / Nt  # prior on t
    ps = torch.ones((1, 1, 1, Ns), device=device) / Ns  # prior on s
    pts = pt * ps  # prior on (t,s)
    pts_x = pts  # X and (T, S) are independent

    # E[n] given that sigmoid parameters are (t,s) and sampling centered on  z
    n_tsz = get_n_from_cos(
        s=lss, theta=lls, target_cos=target_cos,
        delta=delta, d=d).permute(1, 2, 0)
    n_tsz = torch.clamp(n_tsz, max=1e8)  # for numerical stability

    if acq_func == '-E[n]':
        n_ytxsz = n_tsz.reshape(1, Nt, 1, Ns, Nz)

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
    (tt_compute_probs, tt_setting_stats, tt_acq_func,
     tt_max_acquisition, tt_posterior) = 0.0, 0.0, 0.0, 0.0, 0.0

    stop_next = False
    for k in tqdm(range(int(kmax / queries)), desc='bin-search'):
        if stop_next:
            break

        t_start = time.time()

        # Compute some probabilities / expectations
        pyts_x = py_txs * pts_x
        py_x = pyts_x.sum(axis=(1, 3), keepdim=True)
        pt_x = pts_x.sum(axis=3, keepdim=True)
        n_z = (pts_x.reshape(Nt, Ns, 1) * n_tsz).sum(axis=(0, 1))  # E[n | z]
        tt_compute_probs += (time.time() - t_start)
        t_start = time.time()

        # Compute new stats for logs and stopping criterium
        i_ts_max, j_ts_max = unravel_index(pts_x.argmax(), (Nt, Ns))
        ts_max = ttss[:, i_ts_max, j_ts_max].cpu()  # Maximum a posteriori (or prior max)
        ts_map = (pts_x.squeeze() * ttss).sum(axis=(1, 2)).cpu()  # Mean a posteriori (or prior mean)
        iz_tmax = pt_x.argmax().item()
        iz_tmap = int(torch.round((pt_x.squeeze() * ii_t).sum()))  # assumes lin-spaced tt
        iz_best = torch.argmin(n_z).item()
        z_tmax = zz[iz_tmax].item()
        z_tmap = zz[iz_tmap].item()
        z_best = zz[iz_best].item()
        n_zbest_est = n_z[iz_best].item()
        n_ztmap_est = n_z[iz_tmap].item()
        n_ztmax_est = n_z[iz_tmax].item()
        # n_zbest_tru = n_tsz[it_true, is_true, iz_best].item()  # get_n_from_cos(s_, z_best-t_, target_cos, delta, d)
        # n_ztmax_tru = n_tsz[it_true, is_true, iz_best].item()  # get_n_from_cos(s_, z_tmax-t_, target_cos, delta, d)
        # n_ztmap_tru = n_tsz[it_true, is_true, iz_tmax].item()  # get_n_from_cos(s_, z_tmap-t_, target_cos, delta, d)
        tt_setting_stats += time.time() - t_start

        t_start = time.time()
        # Compute acquisition function a(x), x = next sample loc
        if acq_func == 'I(y,t,s)':
            # Compute mutual information I(y, (t, s) | {(xi,yi) : i})
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3))
            Hts = -xlogy(pts_x, pts_x).sum(axis=(0, 1, 3))
            Hyts = -xlogy(pyts_x, pyts_x).sum(axis=(0, 1, 3))
            a_x = Hy + Hts - Hyts  # acquisition = mutual info

        elif acq_func == 'I(y,t)':
            pyt_x = pyts_x.sum(axis=3, keepdim=True)
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3))
            Ht = -xlogy(pt_x, pt_x).sum(axis=(0, 1, 3))
            Hyt = -xlogy(pyt_x, pyt_x).sum(axis=(0, 1, 3))
            a_x = Hy + Ht - Hyt  # acqui = mutual info

        elif acq_func == 'I(y,s)':
            pys_x = pyts_x.sum(axis=1, keepdim=True)
            ps_x = pts_x.sum(axis=1, keepdim=True)
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3))
            Hs = -xlogy(ps_x, ps_x).sum(axis=(0, 1, 3))
            Hys = -xlogy(pys_x, pys_x).sum(axis=(0, 1, 3))
            a_x = Hy + Hs - Hys  # acqui = mutual info

        elif (acq_func == '-E[n]' and
              center_on in {'best', 'near_best'}):
            # a(x) = min_z E[n | x, z] = n | y, x, z
            pts_yx = pyts_x / py_x

            if center_on == 'near_best':
                # a(x) = min_{z near E[t|y,x]} E[n | x, z]
                ps_yx = pts_yx.sum(axis=1, keepdim=True)
                pt_yxs = pts_yx / ps_yx  # TODO: check not NaN
                i_t_yxs = (  # compute index of E[t | y, x, s]
                        ii_t.reshape(1, Nt, 1, 1)  # assumes lin-spaced t
                        * pt_yxs).sum(axis=1, keepdim=True)
                iz_lo = max(int(torch.min(i_t_yxs)) - 1, 0)
                iz_up = int(torch.round(torch.max(
                    (i_t_yxs * ps_yx).sum(axis=3)))) + 1  # E[t | y, x]
            else:
                iz_lo = iz_up = None

            # Compute a(x) = min_z E[n | z, x]
            pts_yxz = pts_yx.reshape(2, Nt, Nx, Ns, 1)
            n_yxz = (pts_yxz * n_ytxsz[..., iz_lo:iz_up]).sum(
                axis=(1, 3), keepdim=True)
            n_yx, _ = torch.min(n_yxz, axis=4)
            n_x = (n_yx * py_x).sum(axis=0).squeeze()  # E[n | x]
            a_x = - n_x

        elif (acq_func == '-E[n]' and
              center_on in {'mode', 'mean'}):
            # a(x) = E_n[n | zj, x]
            #     'mode': with zj = max-likelihood = argmax_t p(t|y,x)
            #     'mean': with zj = E[t|y,x]

            pts_yx = pyts_x / py_x
            pt_yx = pts_yx.sum(axis=3, keepdims=True)
            if center_on == 'mode':  # z = argmax_t p(t | yj, xj)
                iz_yx = torch.argmax(pt_yx, axis=1)  # assumes Nz = Nt
            elif center_on == 'mean':  # z = E[t | y, x]
                iz_yx = torch.round((ii_t.reshape(1, Nt, 1, 1)
                                     * pt_yx).sum(axis=1)).long()
            iz_yx = iz_yx[:, None, :, :, None]
            n_ytxsz, iz_ytxs = torch.broadcast_tensors(n_ytxsz, iz_yx)
            n_ytxs = torch.gather(  # np.take_along_axis(
                n_ytxsz, dim=4, index=iz_ytxs)[..., 0]  # drop last dim (z)
            n_yx = (pts_yx * n_ytxs).sum(axis=(1, 3),
                                         keepdim=True)
            n_x = (n_yx * py_x).sum(axis=0).squeeze()  # E[n | x]
            a_x = - n_x

        else:
            raise ValueError

        # Maximize acquisition function over sampling loc x
        tt_acq_func += time.time() - t_start
        t_start = time.time()
        j_amax = torch.argmax(a_x)
        # xj = xx[j_amax].item()
        j_amax = j_amax.repeat(queries)
        xj = xx[j_amax]
        # yj = int(torch.bernoulli(1-pp[j_amax]))
        yj = torch.bernoulli(1-pp[j_amax]).long()
        # yj, memory = get_model_output(xj, unperturbed, perturbed, decision_function, memory)
        tt_max_acquisition += time.time() - t_start
        t_start = time.time()

        # Update logs
        # vprint(f'E[n]_lim = {n_opt:.2e}\t E[n] = {n_z[j_amax]:.2e}')
        output['xxj'].extend([x.item() for x in xj])
        output['yyj'].extend([y.item() for y in yj])
        output['tts_max'].extend([ts_max] * queries)
        output['tts_map'].extend([ts_map] * queries)
        output['zz_tmax'].extend([z_tmax] * queries)
        output['zz_tmap'].extend([z_tmap] * queries)
        output['zz_best'].extend([z_best] * queries)
        output['nn_best_est'].extend([n_zbest_est] * queries)
        output['nn_tmax_est'].extend([n_ztmax_est] * queries)
        output['nn_tmap_est'].extend([n_ztmap_est] * queries)

        # Test stopping criterion
        if len(output['nn_tmap_est']) > window_size:
            smoothing_kernel = torch.ones(10, ) / 10
            exp_n = torch.tensor(output['nn_tmap_est'][-window_size:])
            diffs = torch.abs(exp_n[1:] - exp_n[:-1])
            if torch.mean(diffs) < 1.:
                stop_next = True

        # Plots
        sq_k = sqrt(k)
        if (plot and (
                (int(sq_k) % 5 == 0 and int(sq_k) - sq_k == 0.)
                or stop_next)):
            plot_acquisition(k, xx, a_x, pts_x, ttss, output, acq_func)

        # Compute posterior (i.e. new prior) for t
        pyj_txjs = py_txs[yj, :, j_amax, :]
        pyj_txjs = pyj_txjs[:, :, None, :].prod(dim=0, keepdim=True)
        pyj_xj = (pyj_txjs * pts_x).sum(axis=(1,3), keepdim=True)
        pts_xyj = pyj_txjs * pts_x / pyj_xj

        # New prior = previous posterior
        pts = pts_xyj
        pts_x = pts  # (t, s) independent of sampling point x
        k += len(xj)
        tt_posterior += time.time() - t_start

    end = time.time()
    vprint(f'Time to finish: {end - start:.2f} s')
    return output


# output = bin_search(
#     acq_func='I(y,t,s)',  # 'I(y,t,s)', 'I(y,t)', 'I(y,s)', '-E[n]'
#     center_on='best',  # only used if acq=-E[n]: 'best', 'near_best', 'mean', 'mode'
#     kmax=1000,  # max number of bin search steps
#     target_cos=.2,  # targeted E[cos(est_grad, true_grad)]
#     delta=.5,  # radius of sphere
#     d=1000,  # input dimension
#     verbose=False,  # print log info
#     eps_=.1,
#     plot=False,
#     grid_size=100)
#
# print()
# print(output['tts_map'][-1])
# print(output['tts_max'][-1])
#
# plt.hist(output['xxj'][:], bins=100)
# plt.xlim(0., 1.)
# plt.show()

