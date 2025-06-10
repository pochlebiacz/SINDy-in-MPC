import numpy as np
from scipy.integrate import solve_ivp


def population(t, state, T=None, u=0):
    """
    Parameters:
    - t: time (unused but included for compatibility with ODE solvers)
    - state: list or array-like [x1, x2]
    - u: control variable
    Returns:
    - derivatives: list of [dx1/dt, dx2/dt]
    """
    population_params = {"a": 0.5, "b": -0.025, "c": -0.5, "d": 0.005}
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
            x0 = population(t=t, state=x0, T=T, u=u[i])
            xk.append(x0)
    else:
        for i in range(len(t)-1):
            x0 = population(t=t, state=x0, T=T)
            xk.append(x0)
    return np.array(xk)


def tracking(t, state, T=None, u=0):
    """
    Parameters:
    - t: time (unused but included for compatibility with ODE solvers)
    - state: list or array-like [x1, x2, x3]
    - u: control variable
    Returns:
    - derivatives: list of [dx1/dt, dx2/dt, dx3/dt]
    """
    x1, x2, x3 = state # x1 = angle of attack, x2 = pitch angle, x3 = pitch rate
    if T is None:
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
    else:
        dx1 = T * (- 0.877 * x1 + x3 - 0.088 * x1 * x3
            + 0.47 * x1**2 - 0.019 * x2**2
            - x1**2 * x3 + 3.846 * x1**3
            - 0.215 * u + 0.28 * x1**2 * u
            + 0.47 * x1 * u**2 + 0.63 * u**3) + x1
        dx2 = T * x3 + x2
        dx3 = T * (- 4.208 * x1 - 0.396 * x3
            - 0.47 * x1**2 - 3.564 * x1**3
            - 20.967 * u + 6.265 * x1**2 * u
            + 46 * x1 * u**2 + 61.1 * u**3) + x3
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


def generate_discrete_tracking_data(t, x0, u=None, T=1):
    assert len(x0) == 3
    xk = [x0]
    
    if u is not None:
        for i in range(len(t)-1):
            x0 = tracking(t=t, state=x0, u=u[i], T=T)
            xk.append(x0)
    else:
        for i in range(len(t)-1):
            x0 = tracking(t=t, state=x0, T=T)
            xk.append(x0)
    return np.array(xk)


def hiv(t, state, T=None, u=0):
    """
    Parameters:
    - t: time (unused but included for compatibility with ODE solvers)
    - state: list or array-like [x1, x2, x3, x4, x5]
    - u: control variable
    Returns:
    - derivatives: list of [dx1/dt, dx2/dt, dx3/dt, dx4/dt, dx5/dt]
    """
    hiv_params = {"lambda": 1, "d": 0.1, "beta": 1, "eta": 0.9799, "a": 0.2, "p1": 1, "p2": 1, 
            "c1": 0.03, "c2": 0.06, "q": 0.5, "b1": 0.1, "b2": 0.01, "h": 0.1}
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

    if T is None:
        dx1 = λ - d * x1 - β * (1 - η * u) * x1 * x2
        dx2 = β * (1 - η * u) * x1 * x2 - a * x2 - p1 * x4 * x2 - p2 * x5 * x2
        dx3 = c2 * x1 * x2 * x3 - c2 * q * x2 * x3 - b2 * x3
        dx4 = c1 * x2 * x4 - b1 * x4
        dx5 = c2 * q * x2 * x3 - h * x5
    else:
        dx1 = T * (λ - d * x1 - β * (1 - η * u) * x1 * x2) + x1
        dx2 = T * (β * (1 - η * u) * x1 * x2 - a * x2 - p1 * x4 * x2 - p2 * x5 * x2) + x2
        dx3 = T * (c2 * x1 * x2 * x3 - c2 * q * x2 * x3 - b2 * x3) + x3
        dx4 = T * (c1 * x2 * x4 - b1 * x4) + x4
        dx5 = T * (c2 * q * x2 * x3 - h * x5) + x5
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


def generate_discrete_hiv_data(t, x0, u=None, T=1):
    assert len(x0) == 5
    xk = [x0]
    
    if u is not None:
        for i in range(len(t)-1):
            x0 = hiv(t=t, state=x0, u=u[i], T=T)
            xk.append(x0)
    else:
        for i in range(len(t)-1):
            x0 = hiv(t=t, state=x0, T=T)
            xk.append(x0)
    return np.array(xk)


def get_reference_population(x0):
    x = np.zeros((200, 2))

    x[0:10, 0] = x0[0]
    x[10:60, 0] = 60
    x[60:120, 0] = 40
    x[120:160, 0] = 45
    x[160:200, 0] = 30

    x[0:10, 1] = x0[1]
    x[10:60, 1] = 30
    x[60:100, 1] = 50
    x[100:160, 1] = 20
    x[160:200, 1] = 40

    return x


def get_reference_tracking(x0):
    x = np.zeros((200, 3))

    x[0:10, 0] = x0[0]
    x[10:60, 0] = -0.1
    x[60:120, 0] = 0.2
    x[120:160, 0] = 0.1
    x[160:200, 0] = -0.2

    x[0:10, 1] = x0[1]
    x[10:60, 1] = 0.1
    x[60:110, 1] = 0.3
    x[110:160, 1] = 0.2
    x[160:200, 1] = 0.4

    x[0:10, 2] = x0[2]
    x[10:60, 2] = 0.2
    x[60:100, 2] = 0.0
    x[100:160, 2] = -0.1
    x[160:200, 2] = 0.1

    return x


def get_reference_hiv(x0):
    x = np.zeros((200, 5))

    x[0:10, 0] = x0[0]
    x[10:40, 0] = 1
    x[40:120, 0] = 3
    x[120:160, 0] = 2
    x[160:200, 0] = 1

    x[0:10, 1] = x0[1]
    x[10:60, 1] = 2
    x[60:100, 1] = 0.5
    x[100:160, 1] = 1
    x[160:200, 1] = 3

    x[0:10, 2] = x0[2]
    x[10:60, 2] = 1.2
    x[60:100, 2] = 1.5
    x[100:160, 2] = 1.3
    x[160:200, 2] = 0.8

    x[0:10, 3] = x0[3]
    x[10:60, 3] = 0.15
    x[60:100, 3] = 0.2
    x[100:160, 3] = 0.1
    x[160:200, 3] = 0.3

    x[0:10, 4] = x0[4]
    x[10:60, 4] = 1.1
    x[60:100, 4] = 1.3
    x[100:160, 4] = 1.2
    x[160:200, 4] = 0.8

    return x
