import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap


def w1_1d(x, y, a=None, b=None):
    """
    exact W1 between weighted 1-D empirical measures
        x: (n,), y: (m,), a: (n,), b: (m,), weights sum to 1 (default uniform)
    W1 = ∫ |F_x - F_y| dt, F_x - F_y is piecewise constant between merged sorted points
    zero weights are allowed, so masked prefixes keep a fixed shape (one compile)
    """
    n, m = x.shape[0], y.shape[0]
    a = jnp.full(n, 1/n, dtype=x.dtype) if a is None else a
    b = jnp.full(m, 1/m, dtype=y.dtype) if b is None else b
    z = jnp.concatenate((x, y))       # (n+m,)
    w = jnp.concatenate((a, -b))      # +a for x, -b for y
    idx = jnp.argsort(z)
    z, w = z[idx], w[idx]
    Fdiff = jnp.cumsum(w)[:-1]        # F_x - F_y on [z_k, z_{k+1})
    return jnp.sum(jnp.abs(Fdiff)*jnp.diff(z))


def gen_sw1(dim, n_proj, key):
    """
    sliced W1, fixed directions θ (n_proj, dim) drawn once
    -> same θ across chain lengths and runs (common random numbers)

    SW1(X, Y) = mean_θ W1(Xθ, Yθ), matches ot.sliced_wasserstein_distance(p=1, projections=θ.T)
    """
    θ = jr.normal(key, (n_proj, dim))
    θ = θ/jnp.linalg.norm(θ, axis=1, keepdims=True)
    vw1 = vmap(w1_1d, in_axes=(1, 1, None, None))
    def sw1(X, Y, a=None, b=None):
        Xθ = X@θ.T # (n, n_proj)
        Yθ = Y@θ.T # (m, n_proj)
        return jnp.mean(vw1(Xθ, Yθ, a, b))
    return sw1, θ


def gen_sw1_prefix(sw1):
    """
    SW1 vs chain length, chains pooled at each length
        X:    (n_chains, N, dim)  chain samples
        Y:    (m, dim)            reference samples
        lens: (k,) int            prefix lengths ≤ N
    returns (k,)
    masked weights instead of X[:, :n] -> one compile for all lens
    lax.map not vmap -> memory O(n_chains*N + m), not k times that
    """
    def sw1_prefix(X, Y, lens):
        C, N, dim = X.shape
        Xf = X.reshape(C*N, dim)
        def one(n):
            mask = (jnp.arange(N) < n).astype(X.dtype)        # (N,)
            a = jnp.tile(mask, C)/(C*n)                        # (C*N,)
            return sw1(Xf, Y, a=a)
        return jax.lax.map(one, lens)
    return sw1_prefix


if __name__ == "__main__":
    import numpy as np
    from scipy.stats import wasserstein_distance
    jax.config.update("jax_enable_x64", True)

    # 1-D exact W1 vs scipy, unequal sizes and weights
    k = jr.split(jr.PRNGKey(0), 4)
    x, y = jr.normal(k[0], (300,)), 0.5 + 2*jr.normal(k[1], (500,))
    a, b = jr.uniform(k[2], (300,)), jr.uniform(k[3], (500,))
    a, b = a/a.sum(), b/b.sum()
    print(f"W1 1-D vs scipy: {float(w1_1d(x, y, a, b)):.12f}  {wasserstein_distance(np.asarray(x), np.asarray(y), np.asarray(a), np.asarray(b)):.12f}")

    # SW1 vs POT, same projections
    dim = 5
    X = jr.normal(k[0], (400, dim))
    Y = jr.normal(k[1], (600, dim)) + 0.3
    sw1, θ = gen_sw1(dim, 200, jr.PRNGKey(7))
    try:
        import ot
        ref = ot.sliced_wasserstein_distance(np.asarray(X), np.asarray(Y), p=1, projections=np.asarray(θ.T))
        print(f"SW1 vs POT:      {float(sw1(X, Y)):.12f}  {float(ref):.12f}")
    except ImportError:
        print("POT not installed, skipping POT cross-check")

    # masked prefix vs explicit slicing
    C, N = 3, 200
    Xc = jr.normal(k[2], (C, N, dim))
    lens = jnp.array([10, 50, 200])
    pre = gen_sw1_prefix(sw1)(Xc, Y, lens)
    sl = jnp.array([sw1(Xc[:, :n].reshape(-1, dim), Y) for n in [10, 50, 200]])
    print(f"prefix vs slice: max |Δ| = {float(jnp.max(jnp.abs(pre - sl))):.2e}")
