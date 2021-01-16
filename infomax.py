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
                                    .5 * torch.sign(x) + .5,
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


def get_alpha(s, theta, delta, d, eps):
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
    return (1. - 2. * eps) * get_alpha_(s_, theta_)


def get_cos_from_n(n, s=float('Inf'), theta=0., delta=1., d=10, eps=0.):
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

    alpha = get_alpha(s, theta, delta, d, eps)
    n = n.to(alpha.device)
    alpha, n = torch.broadcast_tensors(alpha, n)
    ix_nul = (alpha ** 2) == 0.
    ix_pos = ~ix_nul

    out = torch.empty_like(alpha)
    out[ix_nul] = 0.
    out[ix_pos] = 1. / torch.sqrt(1. + (d - 1) / (n[ix_pos] * alpha[ix_pos] ** 2))
    return out


# Nbr of queries n needed to achieve E[cos(est_grad, true_grad)] = target_cos
def get_n_from_cos(target_cos, s=float('Inf'), theta=0., delta=1., d=10, eps=0.):
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

    alpha = get_alpha(s, theta, delta, d, eps)  # returns a tensor
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
    fig1 = axs[1].pcolor(ttss[0], ttss[1], pts_x[0, :, 0, :, 0],
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    axs[1].set_yscale('log')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('s')
    f.colorbar(fig1, ax=axs[1])

    tse_map = output['ttse_map'][-1]
    tse_max = output['ttse_max'][-1]
    n_zbest = output['nn_best_est'][-1]
    f.suptitle('k=%d    En=%.1e    '
               'map=(%4.2f, %4.2f, %4.2f)   '
               'max=(%4.2f, %4.2f, %4.2f)' % (
                   k, n_zbest, *tse_map, *tse_max))
    plt.show()


def get_bernoulli_probs(xx, unperturbed, perturbed, model_interface, true_label, dist_metric='l2'):
    dims = [-1] + [1] * unperturbed.ndim
    xx = xx.view(dims)
    if dist_metric == 'l2':
        batch = (1 - xx) * perturbed + xx * unperturbed
    elif dist_metric == 'linf':
        dist_linf = torch.max(torch.abs(unperturbed - perturbed))
        min_limit = unperturbed - (1-xx) * dist_linf
        max_limit = unperturbed + (1-xx) * dist_linf
        batch = torch.where(perturbed > max_limit, max_limit, perturbed)
        batch = torch.where(batch < min_limit, min_limit, batch)
    probs = model_interface.get_probs_(batch)
    if model_interface.noise == "deterministic":
        pred = probs.argmax(dim=1)
        res = torch.zeros(xx.shape[0], device=batch.device)
        res[pred == true_label] = 1.
    elif model_interface.noise == "stochastic":
        pred = probs.argmax(dim=1)
        res = torch.ones(xx.shape[0], device=batch.device) * model_interface.flip_prob / (model_interface.n_classes - 1)
        res[pred == true_label] = 1 - model_interface.flip_prob
    else:
        res = probs[:, true_label]
    return res



def bin_search(
        unperturbed=None, perturbed=None, model_interface=None,
        acq_func='I(y,t,s,e)', center_on='near_best', kmax=5000, target_cos=.2,
        delta=.5, d=1000, verbose=False, window_size=10, grid_size=100,
        eps_=None, device=None, true_label=None, plot=False, prev_t=None,
        prev_s=None, prev_e=None, prior_frac=1., queries=5,
        tt=None, ss=None, ee=None, stop_criteria="estimate_fluctuation", dist_metric="l2"):
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
        eps_        (float) noise level used (DEPRECATED!!)
        device      (str)   which device to use ('cuda' or 'cpu')
        plot        (bool)  to plot or not to plot
        prev_t      (float) previous estimate of the sigmoid center t (None)
        prev_s      (float) previous estimate of the sigmoid inverse-scale s (None)
        prev_e      (float) previous estimate of the noise eps or nu (None)
        prior_frac  (float) how much to reduce the a priori search interval
                            to the left and to the right of prev_t and prev_s
        queries     (int)   how many queries to perform in each iteration
        tt          (ten)   linear grid where to search the center
        ss          (ten)   logspace grid where to search the inverse-scale s
        ee          (ten)   linear grid where to search the noie level eps

        Using tt, ss or ee disables prev_t, prev_s, prev_e resp.

        Notation conventions in the code:
            * t     center of sigmoid
            * s     inverse scale of sigmoid
            * e     noise levels at +-infty
            * z     centering point for gradient sampling
            * pt_x, pts_x, ...
                    p(t|x), p(t,s|x), ...
            * n_z, n_tsz
                    E[n|z], E[n|t,s,z]
    '''

    t_start = time.time()

    if eps_ is not None:
        raise DeprecationWarning

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
        s_lo = max(log10(prev_s) - prior_frac * 3, -1.)
        s_hi = min(log10(prev_s) + prior_frac * 3, 2.)
        Ns = int(prior_frac * 30) + 1
        # s_lo = log10(prev_s)
        # s_hi = s_lo
        # Ns = 1

    if prev_e is None:
        e_lo, e_hi = 0., .3
        Ne = 7
    else:
        e_lo = prev_e
        e_hi = prev_e
        Ne = 1
        # e_lo = max(prev_e - prior_frac*.3, 0.)
        # e_hi = min(prev_e + prior_frac*.3, .5)
        # Ne = max(int(prior_frac*7) + 1, 3)

    class StoppingCriteria(object):
        def __init__(self, name):
            self.name = name

        def check(self, output, terminated=False):
            if self.name == 'empirical_samples':  # Criteria 1
                if len(output['nn_tmap_est']) > window_size + 1:
                    nn = torch.tensor(output['nn_tmap_est'][-(window_size + 1):])
                    diffs = torch.abs(nn[1:] - nn[:-1])
                    if torch.mean(diffs) < queries or terminated:
                        return True, torch.mean(nn)
            elif self.name == 'expected_samples':  # Criteria 2
                pass
            elif self.name == 'posterior_width':  # Criteria 3
                if len(output['ttse_max']) > window_size:
                    tse = torch.stack(output['ttse_max'][-window_size:])
                    tmax_hi, tmax_lo = max(tse[:, 0]), min(tse[:, 0])
                    nn = [get_n_from_cos(target_cos, theta=tmax_hi - tmax_lo, s=smax, eps=emax, delta=delta, d=d)
                          for (smax, emax) in tse[:, 1:]]
                    n_hi, n_lo = max(nn), min(nn)
                    if abs(n_hi - n_lo) < 1 or terminated:
                        En = get_n_from_cos(target_cos, theta=0.5/grid_size, s=tse[-1, 1], eps=tse[-1, 2],
                                       delta=delta, d=d)
                        return True, max(n_hi, En)
            elif self.name == 'estimate_fluctuation':  # Criteria 4
                if len(output['ttse_max']) > window_size + 1:
                    tse = torch.stack(output['ttse_max'][-(window_size + 1):])
                    tse[:, 1] = torch.log10(tse[:, 1])
                    diffs = torch.abs(tse[1:] - tse[:-1])
                    means = torch.max(diffs, dim=0)[0]
                    if (means[0] <= (t_hi - t_lo) / Nt and means[1] <= (s_hi - s_lo) / Ns \
                            and means[2] <= (e_hi - e_lo) / Ne) or terminated:
                        En = get_n_from_cos(target_cos, theta=1.0/grid_size, s=10.**tse[-1,1], eps=tse[-1,2],
                                            delta=delta, d=d)
                        return True, En
            else:
                raise RuntimeError(f"Unknown Stopping Criteria: {self.name}")
            return False, None

    stopping_criteria = StoppingCriteria(stop_criteria)

    # discretize parameter (search) space
    dtype = torch.float32
    if tt is None:
        tt = torch.linspace(t_lo, t_hi, Nt, dtype=dtype, device=device)
    zz = tt.clone()  # center of sampling ball
    xx = torch.linspace(0., 1., Nx, dtype=dtype, device=device)
    yy = torch.tensor([0, 1], dtype=dtype, device=device)
    if ss is None:
        ss = torch.logspace(s_lo, s_hi, Ns, dtype=dtype, device=device)  # s \in [.01, 100.]
        # ss[-1] = float("Inf")   # xlogy may not work when s is infinite
    if ee is None:
        ee = torch.linspace(e_lo, e_hi, Ne, dtype=dtype, device=device)

    ttssee = torch.stack(torch.meshgrid(tt, ss, ee))  # 2 x Nt x Ns  (numpy indexing='ij')
    ll = zz[:, None] - tt[None, :]  # distance matrix: Nz x Nt
    llse, lsse, lsee = torch.meshgrid(ll.flatten(), ss, ee)
    llse = llse.reshape(Nz, Nt, Ns, Ne)  # Nx x Nt x Ns x Ne
    lsse = lsse.reshape(Nz, Nt, Ns, Ne)  # Nx x Nt x Ns x Ne
    lsee = lsee.reshape(Nz, Nt, Ns, Ne)  # Nx x Nt x Ns x Ne
    ii_t = torch.arange(Nt, device=device)  # indeces of t (useful for later computations)
    if plot:
        ttss = torch.stack(torch.meshgrid(tt, ss))  # 2 x Nt x Ns  (numpy indexing='ij')


    if unperturbed is None:
        pp = get_py_txse(1, t=.3, x=xx, s=3., eps=.1)
    else:
        pp = get_bernoulli_probs(xx, unperturbed, perturbed, model_interface, true_label, dist_metric)

    def vprint(string):
        if verbose:
            print(string)

    start = time.time()

    # Compute likelihood P(y|t,x)
    Y, T, X, S, E = torch.meshgrid(yy, tt, xx, ss, ee)
    py_txse = get_py_txse(Y, T, X, S, E)  # [y, t, x, s, eps] axis always in this order
    pt = torch.ones((1, Nt, 1, 1, 1), device=device) / Nt  # prior on t
    ps = torch.ones((1, 1, 1, Ns, 1), device=device) / Ns  # prior on s
    pe = torch.ones((1, 1, 1, 1, Ne), device=device) / Ne  # prior on e
    ptse = pt * ps * pe # prior on (t,s)
    ptse_x = ptse  # X and (T, S) are independent

    # E[n] given that sigmoid parameters are (t,s) and sampling centered on  z
    n_tsez = get_n_from_cos(
        s=lsse, theta=llse, eps=lsee, target_cos=target_cos,
        delta=delta, d=d).permute(1, 2, 3, 0)  # Nt x Ns x Ne x Nz
    n_tsez = torch.clamp(n_tsez, max=1e8)  # for numerical stability

    if acq_func == '-E[n]':
        n_ytxsz = n_tsz.reshape(1, Nt, 1, Ns, Nz)

    # Initialize logs
    output = {
        'queries_per_loc': [],
        'xxj': [],
        'yyj': [],
        'ttse_max': [],
        'ttse_map': [],
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
    max_queries = queries
    krepeat = int(kmax / max_queries)

    CLIP_MIN = 1e-7
    CLIP_MAX = 1 - 1e-7

    for k in tqdm(range(krepeat), desc='bin-search'):
        if stop_next:
            break
        # if k == krepeat - 1:
        #     stop_next = True

        t_start = time.time()

        queries = min(k // 2 + 1, max_queries)

        # Compute some probabilities / expectations
        ptse_x = torch.clamp(ptse_x, CLIP_MIN, CLIP_MAX)
        ptse_x = ptse_x / ptse_x.sum(axis=(1,3,4), keepdim=True)
        pytse_x = py_txse * ptse_x
        py_x = pytse_x.sum(axis=(1, 3, 4), keepdim=True)
        pts_x = ptse_x.sum(axis=4, keepdim=True)
        pt_x = pts_x.sum(axis=3, keepdim=True)
        n_z = (ptse_x.reshape(Nt, Ns, Ne, 1) * n_tsez).sum(axis=(0, 1, 2))  # E[n | z]
        tt_compute_probs += (time.time() - t_start)
        t_start = time.time()


        # Compute new stats for logs and stopping criterium
        i_tse_max, j_tse_max, h_tse_max = unravel_index(ptse_x.argmax(), (Nt, Ns, Ne))
        tse_max = ttssee[:, i_tse_max, j_tse_max, h_tse_max].cpu()  # Maximum a posteriori (or prior max)
        tse_map = (ptse_x.reshape(Nt, Ns, Ne) * ttssee).sum(axis=(1, 2, 3)).cpu()  # Mean a posteriori (or prior mean)
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
        if acq_func == 'I(y,t,s,e)':
            # Compute mutual information I(y, (t, s, e) | {(xi,yi) : i})
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3, 4))
            Htse = -xlogy(ptse_x, ptse_x).sum(axis=(0, 1, 3, 4))
            Hytse = -xlogy(pytse_x, pytse_x).sum(axis=(0, 1, 3, 4))
            a_x = Hy + Htse - Hytse  # acquisition = mutual info

        elif acq_func == 'I(y,t,s)':
            pyts_x = pytse_x.sum(axis=4, keepdim=True)
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3, 4))
            Hts = -xlogy(pts_x, pts_x).sum(axis=(0, 1, 3, 4))
            Hyts = -xlogy(pyts_x, pyts_x).sum(axis=(0, 1, 3, 4))
            a_x = Hy + Hts - Hyts  # acquisition = mutual info

        elif acq_func == 'I(y,t)':
            pyt_x = pytse_x.sum(axis=(3,4), keepdim=True)
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3, 4))
            Ht = -xlogy(pt_x, pt_x).sum(axis=(0, 1, 3, 4))
            Hyt = -xlogy(pyt_x, pyt_x).sum(axis=(0, 1, 3, 4))
            a_x = Hy + Ht - Hyt  # acqui = mutual info

        elif acq_func == 'I(y,s)':
            pys_x = pytse_x.sum(axis=(1,4), keepdim=True)
            ps_x = pts_x.sum(axis=1, keepdim=True)
            Hy = -xlogy(py_x, py_x).sum(axis=(0, 1, 3, 4))
            Hs = -xlogy(ps_x, ps_x).sum(axis=(0, 1, 3, 4))
            Hys = -xlogy(pys_x, pys_x).sum(axis=(0, 1, 3, 4))
            a_x = Hy + Hs - Hys  # acqui = mutual info

        # TODO: make changes for  eps here!
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

        # TODO: make changes for  eps here!
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
        a_max = torch.max(a_x)
        a_min_to_sample = .9 * a_max if queries > 1 else a_max
        jj_top = torch.where(a_x >= a_min_to_sample)[0]
        j_amax = jj_top[torch.randint(len(jj_top), size=[queries])]

        # # xj = xx[j_amax].item()
        # # yj = int(torch.bernoulli(1-pp[j_amax]))
        # j_amax = torch.argmax(a_x)
        # j_amax = j_amax.repeat(queries)
        xj = xx[j_amax]
        if model_interface is None:
            yj = torch.bernoulli(pp[j_amax]).long()
        else:
            yj = model_interface.sample_bernoulli(pp[j_amax]).long()
        # yj, memory = get_model_output(xj, unperturbed, perturbed, decision_function, memory)
        tt_max_acquisition += time.time() - t_start
        t_start = time.time()

        # Update logs
        # vprint(f'E[n]_lim = {n_opt:.2e}\t E[n] = {n_z[j_amax]:.2e}')
        output['queries_per_loc'].append(queries)
        output['xxj'].extend([x.item() for x in xj])
        output['yyj'].extend([y.item() for y in yj])
        output['ttse_max'].append(tse_max)
        output['ttse_map'].append(tse_map)
        output['zz_tmax'].append(z_tmax)
        output['zz_tmap'].append(z_tmap)
        output['zz_best'].append(z_best)
        output['nn_best_est'].append(n_zbest_est)
        output['nn_tmax_est'].append(n_ztmax_est)
        output['nn_tmap_est'].append(n_ztmap_est)

        # Test stopping criterion
        stop_next, En_ = stopping_criteria.check(output)

        # Plots
        sq_k = sqrt(k)
        if (plot and (
                (int(sq_k) % 5 == 0 and int(sq_k) - sq_k == 0.)
                or stop_next)):
            plot_acquisition(k, xx, a_x, pts_x, ttss, output, acq_func)

        # Compute posterior (i.e. new prior) for t
        pyj_txjse = py_txse[yj, :, j_amax, :, :]
        pyj_txjse = pyj_txjse[:, :, None, :, :].prod(dim=0, keepdim=True)
        pyj_xj = (pyj_txjse * ptse_x).sum(axis=(1,3,4), keepdim=True)
        ptse_xyj = pyj_txjse * ptse_x / pyj_xj

        # New prior = previous posterior
        ptse = ptse_xyj
        ptse_x = ptse  # (t, s, e) independent of sampling point x
        tt_posterior += time.time() - t_start

    end = time.time()
    vprint(f'Time to finish: {end - start:.2f} s')
    # print(tt_compute_probs, tt_setting_stats, tt_acq_func, tt_max_acquisition, tt_posterior)
    if stop_next is False:
        return output, stopping_criteria.check(output, terminated=True)[1]
    else:
        return output, En_


# output = bin_search(
#     acq_func='I(y,t,s,e)',  # 'I(y,t,s,e)', 'I(y,t,s)', 'I(y,t)', 'I(y,s)', '-E[n]'
#     center_on='best',  # only used if acq=-E[n]: 'best', 'near_best', 'mean', 'mode'
#     kmax=1000,  # max number of bin search steps
#     target_cos=.2,  # targeted E[cos(est_grad, true_grad)]
#     delta=.5,  # radius of sphere
#     d=1000,  # input dimension
#     verbose=False,  # print log info
#     plot=True,  # True,
#     grid_size=101,
#     queries=5)
#
# print()
# print(output['tts_map'][-1])
# print(output['tts_max'][-1])
#
# plt.hist(output['xxj'][:], bins=100)
# plt.xlim(0., 1.)
# plt.show()

