"""
Description:
    Numerical integrators for Hamiltonian dynamics.
    USE THE CORRECT ENVIRONMENT:  HMC-Research

Author: John Gallagher
Created: 2026-02-16
Last Modified: 2026-02-16
Version: 0.1
"""
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
from datatypes import QP, IntegratorState, IntegratorConfig
from hamiltonian import J_sym, qJ_sym, pJ_sym, J_sym_flat

def lf_step_qp(
        qp: QP,
        gradH: Callable[[QP], QP],
        τ: float
) -> QP:
    """
    Single lf integration step.

    Does p-first 

    note: that gradH is handled at hamiltonian specification
    lol claude can't do math.
    """
    # Half step momentum
    grad = gradH(qp)
    qhalf= qp.q + 0.5 * τ * grad.symplectic().q
    qhalf_p0 = QP.from_qp(q = qhalf, p = qp.p)

    # Full step position
    grad_half = gradH(qhalf_p0)
    p_new = qp.p + τ * grad_half.symplectic().p
    qhalf_pnew = QP.from_qp(q = qhalf, p = p_new)

    # Half step momentum
    grad_new = gradH(qhalf_pnew)
    q_new = qhalf + 0.5 * τ * grad_new.symplectic().q

    return QP.from_qp(q=q_new, p=p_new)

def lf_step_flat(
    qp_flat: jnp.ndarray,
    gradH_flat: Callable[[jnp.ndarray], jnp.ndarray],
    τ: float
) -> jnp.ndarray:
    """
    Leapfrog step using flat arrays and symplectic operations.
    
    Compatible with your original implementation. 
    **Original implementation should step with p first**
    Uses qJ_sym and pJ_sym for clarity.
    
    Args:
        qp_flat: State as flat array [q, p]
        gradH_flat: Gradient returning flat array
        τ: Step size
        
    Returns:
        Updated flat array
    """
    qp = QP(qp_flat)
    grad_qp = QP(gradH_flat(qp_flat))
    
    # Half momentum step: p -= (τ/2) ∂H/∂q
    qhalf_p0 = qp.to_array() + 0.5 * τ * qJ_sym(grad_qp).to_array()
    
    # Full position step: q += τ ∂H/∂p
    grad_half = QP(gradH_flat(qhalf_p0))
    qhalf_pout = qhalf_p0 + τ * pJ_sym(grad_half).to_array()
    
    # Half momentum step: p -= (τ/2) ∂H/∂q
    grad_out = QP(gradH_flat(qhalf_pout))
    qp_out = qhalf_pout + 0.5 * τ * qJ_sym(grad_out).to_array()
    
    return qp_out

# @partial(jax.jit, static_argnames=['N'])
def lf_integrate(
    qp: QP,
    gradH: Callable[[QP], QP],
    config: IntegratorConfig
) -> QP:
    """
    LF integration using scan.
    
    Args:
        qp: Initial state
        gradH: Hamiltonian gradient
        τ: Step size
        N: Number of steps
        
    Returns:
        Final state after N steps
    """
    def body_fn(qp_state, _):
        qp_new = lf_step_qp(qp_state, gradH, config.τ)
        return qp_new, None
    
    qp_final, _ = jax.lax.scan(body_fn, qp, None, length=config.N)
    return qp_final

def gen_leapfrog(
        gradH: Callable[[QP], QP],
        config: IntegratorConfig
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    gradH: Hamiltonian gradient
    config.τ: Step size
    config.N: Number of steps"""
    def leapfrog_qp(
            qp_in: QP
        ) -> QP:
        """
        LF integration using scan.
        
        Args:
            qp_flat: Initial state            
        Returns:
            Final state after N steps
        """
        def body_fn(qp_state, _):
            qp_new = lf_step_qp(qp_state, gradH, config.τ)
            return qp_new, None
        
        qp_final, _ = jax.lax.scan(body_fn, qp_in, None, length=config.N)
        return qp_final
    return leapfrog_qp

def gen_leapfrog_old(
    gradH_flat: Callable[[jnp.ndarray], jnp.ndarray],
    config: IntegratorConfig
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Generate leapfrog integrator (flat array API).
    
    Compatible with your original gen_leapfrog.
    """
    def leapfrog(qp_flat: jnp.ndarray) -> jnp.ndarray:
        def lf_step(carry_in, _):
            qp0 = carry_in
            qp_out = lf_step_flat(qp0, gradH_flat, config.τ)
            return qp_out, _
        
        qp_final, _ = jax.lax.scan(lf_step, qp_flat, None, length=config.N)
        return qp_final
    
    return leapfrog

def midptNewtonFPI_step(
        qp: QP,
        gradH: Callable[[QP], QP],
        config: IntegratorConfig,
        solve: Callable = jnp.linalg.solve
) -> tuple[QP, IntegratorState]:
    """
    Single implicit midpoint step via FPI (Newton's method)

    Solves Solves: qp_{n+1} = qp_n + τ J ∇H(0.5(qp_n + qp_{n+1}))
    
    Using Newton iteration:
    F(y) = y - qp_n - τ J ∇H(0.5(qp_n + y)) = 0

    x: FPI flat vector 
    y: FPI (Newton) flat vector
    """
    x0 = qp
    
    def G(y):
        """
        Fixed point map: G(y) = x0 + τ J ∇H(0.5(x0 + y))
        """
        midpoint = 0.5 * (x0 + y)
        grad_mid = gradH(midpoint)
        return x0 + config.τ * J_sym(grad_mid)
    
    def F(y):
        """
        I think I need arrays here instead of QP objects like arrays
        """
        return y.x - G(y).x
    
    def newton_step(y):
        jacF = jax.jacobian(F)
        return QP(x0.x - jnp.linalg.solve(jacF(y).x, F(y)))
    
    def cond(carry):
        """bool for while err> tol and iter< max_iter"""
        i, y = carry
        residual = F(y)
        err = jnp.linalg.norm(residual)
        return (err > config.tol) & (i< config.max_iter)
    
    def body_step(carry):
        i, y = carry
        return [i +1, newton_step(y)]
    
    # newton iteration
    n_iter, qp_out = jax.lax.while_loop(cond, body_step, [0, x0])

    residual = F(qp_out)
    res_norm = jnp.linalg.norm(residual)
    state = IntegratorState(
        qp = qp_out,
        residual= QP(residual),
        step_size= config.τ,
        n_iter = n_iter,
        converged = res_norm<= config.tol,
        residual_norm = res_norm
    )
    return qp_out, state

def midptNewtonFPI_integrate(
    qp: QP,
    gradH: Callable[[QP], QP],
    config: IntegratorConfig,
    solve: Callable = jnp.linalg.solve
) -> QP:
    """
    Multi-step implicit midpoint integration via FPI
    Args:
        qp: Initial state
        gradH_flat: Gradient (flat arrays)
        config: Configuration (τ, N, tol, max_iter)
        solve: Linear solver
        
    Returns:
        Final state after N steps
    """
    def body_fn(qp_state, _):
        qp_new, state = midptNewtonFPI_step(qp_state, gradH, config, solve)
        return qp_new, state
    qp_final, states = jax.lax.scan(body_fn, qp, None, length=config.N)
    return qp_final

def gen_midptNewtonFPI(
        gradH: Callable[[QP], QP],
        config: IntegratorConfig,
        solve: Callable = jnp.linalg.solve,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Generate implicit midpt FPI integrator using flat arrays
    """
    def midptFPI_T(qp: QP) -> QP:
        qp_out = midptNewtonFPI_integrate(qp, gradH, config, solve)
        return qp_out
    
    return midptFPI_T

def FPI(g, x0, max_iter, tol):
    x = x0
    err = jnp.inf
    i = 0
    while err > tol and i <=max_iter:
        x_new = g(x)
        err = jnp.linalg.norm(x_new - x)
        x = x_new
        i+=1
    return x

def gauss4(a: float,
           b: float,
            n_pts=4):
    """
    Generate the rescaled gauss quadrature points
    """
    β1 = (b-a)/2
    β0= (a+b)/2
    x0 = -jnp.sqrt(3/7 +(2/7)*jnp.sqrt(6/5))    
    x1 = -jnp.sqrt(3/7 -(2/7)*jnp.sqrt(6/5))    
    x2 = jnp.sqrt(3/7 -(2/7)*jnp.sqrt(6/5))    
    x3 = jnp.sqrt(3/7 +(2/7)*jnp.sqrt(6/5))

    w0 = (18-jnp.sqrt(30))/36
    w1 = (18+jnp.sqrt(30))/36
    w2 = (18+jnp.sqrt(30))/36
    w3 = (18-jnp.sqrt(30))/36

    Xout = β1*jnp.array([x0, x1, x2, x3])+β0
    Wout = β1*jnp.array([w0, w1, w2, w3])
    return Xout, Wout
def gen_AVF(gradH, config):
    ti, wi = gauss4(0,1)
    def AVF(qp0: QP,
            qp1: QP,
            ):
        """Implementing AVF 
        qp_{n+1} = qp_n + τ * J vmap(∇H)((qp_n + qp_{n+1}))

        This is numerical implementation but i'll likely need to do some more solving for gradH_flat
        """
        
        x0 = qp0.to_array()
        X0 = jnp.expand_dims(x0, -1)
        X1 = jnp.expand_dims(qp1.to_array(), -1)
        X = X0+ ti*(X1-X0)
        vgradH = jax.vmap(gradH, in_axes=-1)
        x_out = x0 + config.τ* J_sym_flat(wi@ vgradH(X))
        return QP.from_array(x_out)
    return AVF
def gen_FPI(func, config):
    def FPI(carry, xs):
        xn = carry
        xn1 = func(xn, xn)
        def cond(carry):
            """bool for while err> tol and iter< max_iter"""
            i, x, y = carry
            residual_q = y.q - x.q
            residual_p = y.p - x.p

            err = jnp.sqrt(
                jnp.sum(residual_q**2) +
                jnp.sum(residual_p**2))
            # residual = y.to_array()-x.to_array()
            # err = jnp.linalg.norm(residual)
            return (err > config.tol) & (i< config.max_iter)
        def body_step(carry):
            i, x, y = carry
            return (i +1, y, func(xn, y))
        n_iter, _, x_next = jax.lax.while_loop(cond, body_step, (1, xn, xn1))
        return x_next, x_next
    return FPI
def gen_AVF_FPI_T(F_flat, config):
    AVF_FPI = gen_FPI(gen_AVF(F_flat, config), config)
    def AVF_FPI_T(θω_init):
        final_state, trajectory = jax.lax.scan(
            AVF_FPI,
            θω_init,
            None,
            length=config.N,
        )
        return trajectory
    return AVF_FPI_T