import jax
import jax.numpy as jnp
import jax.random as jr
from datatypes import QP


def gen_funnel(dim_q, σ=3.0):
    """
    Neal's funnel, q = (v, x_1, ..., x_{dim_q-1})
        v ~ N(0, σ²),  x_i | v ~ N(0, e^v)

    U(q) = v²/(2σ²) + e^{-v} ‖x‖²/2 + (dim_q-1) v/2
    H(q,p) = U(q) + ‖p‖²/2     (M = I)

    returns U, H_flat, gradH, sample, var_true
        U:        (dim_q,) -> scalar
        H_flat:   (2*dim_q,) -> scalar
        gradH:    QP -> QP, matches gen_AVF
        sample:   (key, n) -> (n, dim_q), exact draws
        var_true: (dim_q,), [σ², e^{σ²/2}, ..., e^{σ²/2}]
    """
    d = dim_q - 1
    def U(q):
        v, x = q[0], q[1:]
        return v**2/(2*σ**2) + 0.5*jnp.exp(-v)*(x@x) + 0.5*d*v
    def H_flat(z):
        q, p = z[:dim_q], z[dim_q:]
        return U(q) + 0.5*(p@p)
    def gradH(qp: QP) -> QP:
        return QP(jax.grad(H_flat)(qp.x)) # (2*dim_q,), QP in, QP out
    def sample(key, n):
        kv, kx = jr.split(key)
        v = σ*jr.normal(kv, (n, 1))                 # (n, 1)
        x = jnp.exp(v/2)*jr.normal(kx, (n, d))      # (n, d)
        return jnp.hstack((v, x))                   # (n, dim_q)
    var_true = jnp.concatenate((jnp.array([σ**2]), jnp.exp(σ**2/2)*jnp.ones(d)))
    return U, H_flat, gradH, sample, var_true


def gradU_funnel(q, σ=3.0):
    """analytic ∇U, independent reference for gradH"""
    v, x = q[0], q[1:]
    d = x.shape[0]
    dv = v/σ**2 - 0.5*jnp.exp(-v)*(x@x) + 0.5*d
    dx = jnp.exp(-v)*x
    return jnp.concatenate((jnp.array([dv]), dx))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    dim_q, σ = 10, 3.0
    U, H_flat, gradH, sample, var_true = gen_funnel(dim_q, σ)

    # ∇U: autodiff vs analytic
    q = jr.normal(jr.PRNGKey(1), (dim_q,))
    p = jr.normal(jr.PRNGKey(2), (dim_q,))
    g = gradH(QP(jnp.concatenate((q, p))))
    err_q = jnp.max(jnp.abs(g.q - gradU_funnel(q, σ)))
    err_p = jnp.max(jnp.abs(g.p - p))
    print(f"‖∇U autodiff - analytic‖∞ = {err_q:.2e}")
    print(f"‖∇K - p‖∞                 = {err_p:.2e}")

    # exact sampler moments
    X = sample(jr.PRNGKey(0), 1_000_000)
    print(f"mean v = {X[:,0].mean():+.4f}   (0)")
    print(f"var  v = {X[:,0].var():.4f}   ({σ**2})")
    print(f"var  x = {X[:,1:].var(axis=0).mean():.2f}   ({float(var_true[1]):.2f}, heavy tailed, slow to converge)")
