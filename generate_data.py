import numpy as np
from scipy.integrate import solve_ivp
import json

with open('params.json', 'r') as f:
    population_params = json.load(f)["population"]

with open('params.json', 'r') as f:
    hiv_params = json.load(f)["hiv"]


def population(t, state, T=None, u=0):
    """
    Parameters:
    - t: time (unused but included for compatibility with ODE solvers)
    - state: list or array-like [x1, x2]
    - u: control variable
    Returns:
    - derivatives: list of [dx1/dt, dx2/dt]
    """
    x1, x2 = state # x1 = prey, x2 = predator

    a = population_params["a"]
    b = population_params["b"]
    c = population_params["c"]
    d = population_params["d"]
    if T is None:
        dx1 = a * x1 + b * x1 * x2
        dx2 = c * x2 + d * x1* x2 + u
    else:
        dx1 = T * (a * x1 + b * x1 * x2) + x1
        dx2 = T * (c * x2 + d * x1* x2 + u) + x2
    return [dx1, dx2]


def generate_population_data(t, x0, u=None):
    assert len(x0) == 2
    
    if u is not None:
        x = solve_ivp(lambda t, y: population(t=t, state=y, u=u(t)), (t[0], t[-1]), x0, t_eval=t, method='LSODA').y.T
        x_dot = np.array([population(time_step, state, u=u(time_step)) for time_step, state in zip(t, x)])
    else:
        x = solve_ivp(lambda t, y: population(t=t, state=y), (t[0], t[-1]), x0, t_eval=t, method='LSODA').y.T
        x_dot = np.array([population(time_step, state) for time_step, state in zip(t, x)])
    return x, x_dot


def generate_discrete_population_data(t, x0, u=None, T=1):
    assert len(x0) == 2
    xk = [x0]
    
    if u is not None:
        for i in range(len(t)-1):
            x0 = population(t=t, state=x0, u=u[i], T=T)
            xk.append(x0)
    else:
        for i in range(len(t)-1):
            x0 = population(t=t, state=x0, T=T)
            xk.append(x0)
    return np.array(xk)


def tracking(t, state, u=0):
    """
    Parameters:
    - t: time (unused but included for compatibility with ODE solvers)
    - state: list or array-like [x1, x2, x3]
    - u: control variable
    Returns:
    - derivatives: list of [dx1/dt, dx2/dt, dx3/dt]
    """
    x1, x2, x3 = state # x1 = angle of attack, x2 = pitch angle, x3 = pitch rate 
    dx1 = (- 0.877 * x1 + x3 - 0.088 * x1 * x3
           + 0.47 * x1**2 - 0.019 * x2**2
           - x1**2 * x3 + 3.846 * x1**3
           - 0.215 * u + 0.28 * x1**2 * u
           + 0.47 * x1 * u**2 + 0.63 * u**3)
    dx2 = x3
    dx3 = (- 4.208 * x1 - 0.396 * x3
           - 0.47 * x1**2 - 3.564 * x1**3
           - 20.967 * u + 6.265 * x1**2 * u
           + 46 * x1 * u**2 + 61.1 * u**3)

    return [dx1, dx2, dx3]


def generate_tracking_data(t, x0, u=None):
    assert len(x0) == 3
    
    if u is not None:
        x = solve_ivp(lambda t, y: tracking(t=t, state=y, u=u(t)), (t[0], t[-1]), x0, t_eval=t, method='LSODA').y.T
        x_dot = np.array([tracking(time_step, state, u=u(time_step)) for time_step, state in zip(t, x)])
    else:
        x = solve_ivp(lambda t, y: tracking(t=t, state=y), (t[0], t[-1]), x0, t_eval=t, method='LSODA').y.T
        x_dot = np.array([tracking(time_step, state) for time_step, state in zip(t, x)])
    return x, x_dot


def hiv(t, state, u=0):
    """
    Parameters:
    - t: time (unused but included for compatibility with ODE solvers)
    - state: list or array-like [x1, x2, x3, x4, x5]
    - u: control variable
    Returns:
    - derivatives: list of [dx1/dt, dx2/dt, dx3/dt, dx4/dt, dx5/dt]
    """
    x1, x2, x3, x4, x5 = state # x1 = healthy CD4+ T cells, x2 = HIV-infected CD4+ T cells, 
                               # x3 = CLT precursors, x4 = helper-independent CTLs, x5 = helper-dependent CTLs
    λ = hiv_params["lambda"]
    d = hiv_params["d"]
    β = hiv_params["beta"]
    η = hiv_params["eta"]
    a = hiv_params["a"]
    p1 = hiv_params["p1"]
    p2 = hiv_params["p2"]
    c1 = hiv_params["c1"]
    c2 = hiv_params["c2"]
    q = hiv_params["q"]
    b1 = hiv_params["b1"]
    b2 = hiv_params["b2"]
    h = hiv_params["h"]

    dx1 = λ - d * x1 - β * (1 - η * u) * x1 * x2
    dx2 = β * (1 - η * u) * x1 * x2 - a * x2 - p1 * x4 * x2 - p2 * x5 * x2
    dx3 = c2 * x1 * x2 * x3 - c2 * q * x2 * x3 - b2 * x3
    dx4 = c1 * x2 * x4 - b1 * x4
    dx5 = c2 * q * x2 * x3 - h * x5

    return [dx1, dx2, dx3, dx4, dx5]


def generate_hiv_data(t, x0, u=None):
    assert len(x0) == 5
    
    if u is not None:
        x = solve_ivp(lambda t, y: hiv(t=t, state=y, u=u(t)), (t[0], t[-1]), x0, t_eval=t, method='LSODA').y.T
        x_dot = np.array([hiv(time_step, state, u=u(time_step)) for time_step, state in zip(t, x)])
    else:
        x = solve_ivp(lambda t, y: hiv(t=t, state=y), (t[0], t[-1]), x0, t_eval=t, method='LSODA').y.T
        x_dot = np.array([hiv(time_step, state) for time_step, state in zip(t, x)])
    return x, x_dot