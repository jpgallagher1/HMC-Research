"""Standalone Hudson Bay posterior; does not execute notebook cells."""
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

PARAMETERS = ('alpha', 'beta', 'gamma', 'delta', 'hare0', 'lynx0', 'sigma')
DEFAULT_DATA = Path(__file__).resolve().parents[3] / 'hudson-bay-lynx-hare.csv'


def load_data(path=DEFAULT_DATA):
    data = np.loadtxt(path, delimiter=',', skiprows=3)
    if data.ndim != 2 or data.shape[1] != 3 or not np.all(np.isfinite(data)):
        raise ValueError('Expected finite year, lynx, hare columns')
    if np.any(data[:, 1:] <= 0) or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('Observation counts must be positive and years increasing')
    return data[:, 0] - data[0, 0], data[:, 2], data[:, 1]


def build_model(data_path=DEFAULT_DATA, ode_dt=.02, notebook_priors=False):
    if ode_dt <= 0:
        raise ValueError('ode_dt must be positive')
    ts, hare, lynx = map(jnp.asarray, load_data(data_path))
    mean = jnp.array([1., .05, 1., .05, jnp.log(10.), jnp.log(10.), -1.])
    sd = jnp.array([.5, .05, .5, .05, 1., 1., 1.])

    def rhs(t, y, args):
        x, z = y
        a, b, g, d = args
        return a*x - b*x*z, -g*z + d*x*z

    def predict(phi):
        theta = jnp.exp(phi)
        sol = diffeqsolve(ODETerm(rhs), Tsit5(), t0=ts[0], t1=ts[-1],
                         dt0=ode_dt, y0=(theta[4], theta[5]), args=theta[:4],
                         saveat=SaveAt(ts=ts), max_steps=100000, throw=False)
        return sol.ys

    def logposterior(phi):
        theta = jnp.exp(phi)
        prey, predator = predict(phi)
        ll = norm.logpdf(jnp.log(hare), jnp.log(prey), theta[-1]).sum()
        ll += norm.logpdf(jnp.log(lynx), jnp.log(predator), theta[-1]).sum()
        # Explicit legacy switch makes the notebook's prior bugs reproducible.
        normal_sd = mean[:4] if notebook_priors else sd[:4]
        lognormal_mean = sd[4:] if notebook_priors else mean[4:]
        lp = norm.logpdf(theta[:4], mean[:4], normal_sd).sum() + phi[:4].sum()
        lp += norm.logpdf(phi[4:], lognormal_mean, sd[4:]).sum()
        return ll + lp

    center = jnp.concatenate((jnp.log(mean[:4]), mean[4:]))
    return logposterior, predict, center
