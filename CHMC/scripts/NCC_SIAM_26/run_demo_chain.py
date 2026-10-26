# SANDBOX DRIVER: AVF-Newton CHMC on the funnel with log|det J| in the acceptance, saves npz for plots.py
import sys
from types import SimpleNamespace
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from datatypes import QP, J_sym
import avf_logdet as M
M.QP, M.J_sym = QP, J_sym
from funnel import gen_funnel

dim_q, C, n_samp = 5, 4, 2000
U, H_flat, gradH, sample, var_true = gen_funnel(dim_q)

def gen_chain(gen_T, cfg):
    T = gen_T(gradH, cfg)
    def step(q, key):
        kp, ku = jr.split(key)
        p = jr.normal(kp, q.shape)
        z0 = QP(jnp.concatenate((q, p)))
        z1, ld = T(z0)
        dH = H_flat(z1.x) - H_flat(z0.x)
        acc = M.accept_reject(dH, ku, ld)
        return jnp.where(acc, z1.q, q), (jnp.where(acc, z1.q, q), ld, acc)
    def chain(key):
        _, (S, lds, A) = jax.lax.scan(step, jnp.zeros(dim_q), jr.split(key, n_samp))
        return S, lds, A
    return jax.jit(jax.vmap(chain))

base = dict(τ=0.1, n_pts=5, tol=1e-10, max_iter=50, N=15, debug=False, logdet=True,
            gen_gauss=False, AA_beta=1.0, AA_m=5)
for name, gen_T in [("newton", M.gen_AVF_NewtonFPI_T), ("aa", M.gen_AVF_AA_T)]:
    S, lds, A = gen_chain(gen_T, SimpleNamespace(**base))(jr.split(jr.PRNGKey(3), C))
    np.savez(f"run_{name}.npz", samples=np.asarray(S), log_dets=np.asarray(lds),
             accept=np.asarray(A), label=f"AVF {name} τ={base['τ']} L={base['N']}")
    print(f"{name}: accept={float(A.mean()):.3f}  log_det range=[{float(jnp.nanmin(lds)):.3e}, {float(jnp.nanmax(lds)):.3e}]  nonfinite={int(jnp.sum(~jnp.isfinite(lds)))}  var v={float(S[...,0].var()):.2f}")
