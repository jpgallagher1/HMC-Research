"""Rank-normalized bulk/tail ESS and folded split R-hat, with no ArviZ dependency."""
import numpy as np
from scipy.stats import rankdata, norm
from numpyro.diagnostics import effective_sample_size, gelman_rubin


def split_chains(x):
    half = x.shape[1] // 2
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def rank_normalize(x):
    flat = x.reshape(-1, x.shape[-1])
    ranks = np.stack([rankdata(flat[:, j]) for j in range(flat.shape[1])], axis=-1)
    return norm.ppf((ranks - .375) / (len(flat) + .25)).reshape(x.shape)


def diagnose(phi):
    """Input (chains, post-warmup draws, parameters); rejected states remain included."""
    x = np.asarray(phi, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 8:
        raise ValueError('Need (chains>=2, draws>=8, parameters)')
    d = x.shape[-1]
    bad = (~np.isfinite(x).all(axis=(0, 1))) | (np.var(x, axis=(0, 1)) == 0)
    # Every chain must move; constant-but-different chains cannot yield useful ESS.
    bad |= np.any(np.var(x, axis=1) == 0, axis=0)
    bulk, tail, rhat, mean_ess = (np.full(d, np.nan) for _ in range(4))
    good = ~bad
    if np.any(good):
        y = split_chains(x[..., good])
        z = rank_normalize(y)
        folded = rank_normalize(np.abs(y - np.median(y, axis=(0, 1))))
        bulk[good] = effective_sample_size(z)
        rhat[good] = np.maximum(gelman_rubin(z), gelman_rubin(folded))
        lo, hi = np.quantile(y, [.05, .95], axis=(0, 1))
        tail[good] = np.minimum(effective_sample_size((y <= lo).astype(float)),
                                effective_sample_size((y >= hi).astype(float)))
        mean_ess[good] = effective_sample_size(split_chains(np.exp(x[..., good])))
    return dict(ess_bulk=bulk, ess_tail=tail, rhat=rhat,
                ess_mean_theta=mean_ess, stuck_parameter=bad)


def efficiency(phi, sample_seconds, warmup_seconds=0., rhat_limit=1.01):
    if sample_seconds <= 0:
        raise ValueError('sampling time must be positive')
    diagnostics = diagnose(phi)
    valid = all(np.all(np.isfinite(diagnostics[k])) for k in ('ess_bulk', 'ess_tail', 'rhat'))
    valid &= bool(np.all(diagnostics['rhat'] <= rhat_limit))
    minimum = float(np.min(diagnostics['ess_bulk'])) if valid else float('nan')
    return {**diagnostics, 'converged': valid, 'min_bulk_ess': minimum,
            'seconds_per_min_bulk_ess': sample_seconds/minimum if valid else float('nan'),
            'total_seconds_per_min_bulk_ess': (sample_seconds+warmup_seconds)/minimum if valid else float('nan')}
