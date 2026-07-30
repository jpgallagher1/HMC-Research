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
from jax import vmap
import jax.numpy as jnp
from functools import partial
from typing import Callable
from datatypes import QP, IntegratorState, IntegratorConfig
from hamiltonian import J_sym, qJ_sym, pJ_sym, J_sym_flat
from scipy.special import roots_legendre
from jaxopt import AndersonAcceleration

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
        def body_fn(carry, _):
            qp_state = carry
            qp_new = lf_step_qp(qp_state, gradH, config.τ)
            return qp_new, qp_new
        
        qp_final, traj = jax.lax.scan(body_fn, qp_in, None, length=config.N)
        if config.debug:
            return qp_final, traj
        else:
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
    returns [QP, IntegratorState]
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
    """
    ****USES gradH*****
    """
    roots, weights = roots_legendre(config.n_pts)
    a, b = 0, 1
    β1 = (b-a)/2
    β0 = (a+b)/2
    ti = β1*roots+β0
    wi = β1*weights

    def AVF(qp0: QP,
            qp1: QP,
            )-> QP:
        """Implementing AVF 
        qp_{n+1} = qp_n + τ * J vmap(∇H)((qp_n + qp_{n+1}))

        This is numerical implementation but i'll likely need to do some more solving for gradH_flat
        """
        X0 = jnp.expand_dims(qp0.x, 0) # (1, 2*dim)
        X1 = jnp.expand_dims(qp1.x, 0) # (1, 2*dim)
        X = X0+ ti[:,None]@(X1-X0) # (nodes, 2*dim) + (nodes, 1)(1, 2*dim)
        gradX = vmap(gradH)(QP(X)) #QP in, QP out (nodes, 2*dim)
        x_out = qp0 + config.τ* J_sym(QP(wi@ gradX.x)) # Is the right implementation to do vgrad(X).x or vgrad(X)? 
        return x_out
    return AVF
def gen_FPI(func, config):
    """
    Possible todo: update the carry to use the IntegratorState NamedTuple
    """
    def FPI(carry, xs):
        xn = carry
        xn1 = func(xn, xn)
        def cond(carry):
            """bool for while err> tol and iter< max_iter"""
            i, x, y = carry
            err = jnp.linalg.norm(y.x-x.x)
            # err = jnp.linalg.norm(residual)
            return (err > config.tol) & (i< config.max_iter)
        def body_step(carry):
            i, x, y = carry
            return (i+1, y, func(xn, y))
        n_iter, x_last, x_next = jax.lax.while_loop(cond, body_step, (1, xn, xn1))
        resid = jnp.linalg.norm(x_next.x - x_last.x)
        return x_next, (x_next, n_iter, resid)     # scan now iters & residual per step
    return FPI

def gen_AVF_FPI_T(gradH_flat, config):
    """
    debug returns all scan outputs: 
        final_state, (traj, iters, resids)
    """
    AVFfunc = gen_AVF(gradH_flat, config)
    AVF_FPI = gen_FPI(AVFfunc, config)
    def AVF_FPI_T(θω_init):
        final_state, (traj, iters, resids)  = jax.lax.scan(
            AVF_FPI,
            θω_init,
            None,
            length=config.N,
        )
        if config.debug:
            return final_state, traj, iters, resids 
        else:
            return final_state
    return AVF_FPI_T

def gen_AVF_AA_T(gradH_flat, config):
    """
    debug returns all scan outputs: 
        final_state, (traj, iters, resids)
    """
    AVF = gen_AVF(gradH_flat, config)
    # jaxopt AndersonAcceleration does FPI in the first entry
    AVF_rev = lambda y,x0: AVF(x0, y)
    
    AVF_AA = AndersonAcceleration(AVF_rev,beta=config.AA_beta,history_size=config.AA_m, tol=config.tol,implicit_diff=True,maxiter=config.max_iter)
    def AVF_AA_t(carry, _):
        """
        Body Function for scan
        AA for one dt, timestep. 
        """
        init_guess = AVF(carry, carry) 
        out = AVF_AA.run(init_guess, carry)
        return out.params, (out.params, out.state.iter_num)

    def AVF_AA_T(θω_init):
         
        final_state, (traj, iters)  = jax.lax.scan(AVF_AA_t, 
                                                           θω_init, 
                                                           None, 
                                                           config.N
                                                           )
        if config.debug:
            return final_state, traj, iters
        else:
            return final_state
    return AVF_AA_T



def gen_AVF_vecs(config):
    """
    
    evaluation points for the AVF scheme
    qp_{n+1} = qp_n + τ * J vmap(∇H)((qp_n + qp_{n+1}))
    gives the vectors at the shifted lagrang nodes
    for use in the vectorized gradient and jacobian. 
    """
    roots, weights = roots_legendre(config.n_pts)
    a, b = 0, 1
    β1 = (b-a)/2
    β0 = (a+b)/2
    ti = β1*roots+β0
    wi = β1*weights
    def AVF_vecs(qp0: QP,
                 qp1: QP,
                )-> QP:
        """
        in qp0, qp1
        Out: X, wi, ti
        """
        X0 = jnp.expand_dims(qp0.x, -1)
        X1 = jnp.expand_dims(qp1.x, -1)
        X = X0+ ti*(X1-X0)
        # apply QP(X.T) to have X.q and X.p behave correctly: 
        # X.q 
        return QP(X), wi, ti
    return AVF_vecs

def gen_gradH_gen_gauss(β):
    """
    H(x) =  1/β * q**β + 0.5*p**2
    NO SYMPLECTIC MATRIX APPLIED
    """
    def gradH(qp):
        """
        H(x) =  1/β * q**β + 0.5*p**2
        ∇H =  [q**(β-1), p]
        """
        return QP(jnp.concat([qp.q**(β-1), qp.p]))
    return gradH
def gen_jacH_gengauss(β):
    """
    H(x) =  1/β * q**β + 0.5*p**2
    NO SYMPLECTIC MATRIX APPLIED
    """
    def jacH(qp):
        """
        diagonals only for separable hamiltonian of the form 
        H(x) =  1/β * q**β + 0.5*p**2
        with 
        ∇**2 H =  [[(β-1)*q**(β-2), 0], 
                            [0 , 1]]
        """
        return QP(jnp.concat([(β-1)*qp.q**(β-2),jnp.ones_like(qp.p)]))
    return jacH
# check that the jacH makes the same jacobian as jacfwd(AVF)
def gen_AVF_jacF_diag(β, config):
    """
    Inputs: β, config
    returns {τ J_sym @ Σ w_i *t_i * ∇**2 H (y)}

    For separable hamiltonian of the form 
    H(x) =  1/β * q**β + 0.5*p**2
    with 
    ∇**2 H =  [[(β-1)*q**(β-2), 0], 
                        [0 , 1]]
    F(y) = y - AVF(x0, y)
    J_F(y) = I - {τ J_sym @ Σ w_i *t_i * ∇**2 H (y)}
    returns the parenthesis portion as a single vector (the off diagonal component)
    """

    AVF_vecs = gen_AVF_vecs(config)
    jacH_gg = gen_jacH_gengauss(β)
    vjacH_gg = vmap(jacH_gg, in_axes=-1)


    def jacF_AVF(x0, y):
        """
        returns {τ J_sym @ Σ w_i *t_i * ∇**2 H (y)}
        of the approx form [[A, B], [C, D]] = JacF
        where qp_out.q = diag(B), and qp_out.p = diag(C)
        
        
        For separable hamiltonian of the form 
        H(x) =  1/β * q**β + 0.5*p**2
        with 
        ∇**2 H =  [[(β-1)*q**(β-2), 0], 
                            [0 , 1]]
        F(y) = y - AVF(x0, y)
        J_F(y) = I - {τ J_sym @ Σ w_i *t_i * ∇**2 H (y)}
        returns the bracket portion as a single vector (the off diagonal component)
        """
        X, wi, ti = AVF_vecs(x0, y)
        # X shape: (2n, config.npts) QP object
        # wi shape: (config.npts,)
        # ti shape: (config.npts,)
        D_AVF = jnp.sum(wi*ti*vjacH_gg(X).x.T, axis=-1) 
        qp_out = -config.τ*J_sym(QP(D_AVF))
        return qp_out
    return jacF_AVF

# If [[a,b],[c,d]] = A, and by
# then I have diag(b) and diag(c) and a, d = jnp.eye(dim)
# I can solve in multiple ways: or I can u

def solve2x2scalar(a,b,c,d,e,f):
    """
    this is the fastest solve for 2x2 Ax=b
    [a,b],[c,d] = A
    e,f = by
    """
    x2 = (a*f - e*c)/(a*d-b*c)
    x1 = (e*d-b*f)/(a*d-b*c)
    return x1, x2

def gen_AVF_NewtonFPI_T(gradH_flat, config, β=4):
    """
    optional string: solve='gen_gauss'
    then used for generalized-p-gauss
    """
    AVF = gen_AVF(gradH_flat, config)
    def cond(carry):
        i, _, dnorm = carry
        return (dnorm > config.tol) & (i < config.max_iter)
    if config.gen_gauss:
        jacF_x0_y = gen_AVF_jacF_diag(β, config)
        vsol2x2 = vmap(solve2x2scalar, in_axes=(None, 0, 0, None, 0,0))
        def AVF_newton_t(qp:QP, _)-> QP:
            x0 = qp
            def F(y:QP): 
                return y - AVF(x0, y)   # flat residual R^n->R^n
            jacF = lambda y: jacF_x0_y(x0, y) # QP object

            def body(carry):
                i, y, _ = carry # array or QP object?
                Fyx = F(y)
                diagjacFyx = jacF(y)
                dq, dp = vsol2x2(1, diagjacFyx.q, diagjacFyx.p, 1, Fyx.q, Fyx.p)
                dy = QP(jnp.hstack((dq,dp)))
                return i+1, y-dy, jnp.linalg.norm(dy.x)
            n_iter, x_next, residual = jax.lax.while_loop(cond, body, (0, x0, jnp.inf))
            return x_next, (n_iter, x_next, residual)
    else:
        def AVF_newton_t(qp: QP, _) -> QP:
            x0 = qp
            def F(yx): 
                return (QP(yx) - AVF(x0, QP(yx))).x   # flat residual R^n->R^n
            jacF = jax.jacfwd(F)                             # square system


            def body(carry):
                i, yx, _ = carry
                dy = jnp.linalg.solve(jacF(yx), F(yx))     # Newton increment, this step
                return i + 1, yx - dy, jnp.linalg.norm(dy)

            n_iter, x_next, residual = jax.lax.while_loop(cond, body, (0, x0.x, jnp.inf))
            return  QP(x_next), (n_iter, x_next, residual)
    def integrate_T(qp_init):
        def Newton_step(carry, _):
            next = AVF_newton_t(carry, _)
            return next
        last, (iters, traj, resids) = jax.lax.scan(Newton_step, qp_init, None, config.N)
        if config.debug:
            return last, traj, iters, resids 
        else:
            return last
    return integrate_T