import numpy as np
from scipy.integrate import solve_ivp
from decimal import Decimal
import matplotlib.pyplot as plt
from pysindy.feature_library import PolynomialLibrary, FourierLibrary
import pysindy as ps
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


def get_population_model(Ts=0.1, thr=1e-4, deg=2):
    x0 = [50, 50]
    t = np.linspace(0, 100, int(100/Ts)+1)
    u = lambda t : np.sin(t)

    xk = generate_discrete_population_data(t, x0, T=Ts, u=u(t))

    model = ps.SINDy(
        feature_library=PolynomialLibrary(degree=deg),
        # feature_library=FourierLibrary(n_frequencies=2),
        optimizer=ps.STLSQ(threshold=thr),
        feature_names=[f'x{i+1}' for i in range(len(x0))]+['u'],
        discrete_time=True)
    
    model.fit(x=xk, u=u(t))
    model.print(precision=int(3 - np.log10(Ts)))

    return model


def get_reference_tracking(x0):
    x = np.zeros((200, 3))

    x[0:10, 0] = x0[0]
    x[10:40, 0] = -0.1
    x[40:120, 0] = 0.2
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


def get_tracking_model(Ts=0.1, thr=1e-5, deg=3):
    x0 = [-0.1, 0.2, -0.1]
    t = np.linspace(0, 20, int(20/Ts)+1)
    u = lambda t : 0.001*np.sin(t)

    xk = generate_discrete_tracking_data(t, x0, T=Ts, u=u(t))

    model = ps.SINDy(
        feature_library=PolynomialLibrary(degree=deg),
        # feature_library=FourierLibrary(n_frequencies=2),
        optimizer=ps.STLSQ(threshold=thr),
        feature_names=[f'x{i+1}' for i in range(len(x0))]+['u'],
        discrete_time=True)
    
    model.fit(x=xk, u=u(t))
    model.print(precision=int(3 - np.log10(Ts)))

    return model


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


def get_hiv_model(Ts=0.1, thr=1e-4, deg=3):
    x0 = [1, 1, 1, 1, 1]
    t = np.linspace(0, 100, int(100/Ts)+1)
    u = lambda t : 0.1*np.sin(t)

    xk = generate_discrete_hiv_data(t, x0=x0, T=Ts, u=u(t))

    model = ps.SINDy(
        feature_library=PolynomialLibrary(degree=deg),
        # feature_library=FourierLibrary(n_frequencies=2),
        optimizer=ps.STLSQ(threshold=thr),
        feature_names=[f'x{i+1}' for i in range(len(x0))]+['u'],
        discrete_time=True)
    
    model.fit(x=xk, u=u(t))
    model.print(precision=int(3 - np.log10(Ts)))

    return model


def linearize(model, x, u, eps=1e-6):
    x = np.array(x, dtype=float)
    u = float(u)
    nx = len(x)

    f0 = model.simulate(x, t=2, u=np.array([u]))[-1]
    A = np.zeros((nx, nx))
    B = np.zeros((nx, 1))

    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        xp = model.simulate(x+dx, t=2, u=np.array([u]))[-1]
        A[:, i] = (xp - f0) / eps

    up = model.simulate(x, t=2, u=np.array([u+eps]))[-1]
    B[:, 0] = (up - f0) / eps

    return A, B


def plot(y_zad, x, u, w, save=False):
    fig, axs = plt.subplots(len(x[0]), 1, figsize=(10, 2.4*len(x[0])))
    if len(x[0]) == 2:
        axs[0].set_title("Liczba osobników")
        axs[0].set_ylabel("Populacja ofiar")
        axs[1].set_ylabel("Populacja drapieżników")
        axs[-1].set_xlabel("Czas [dni]")
        process = 1
    elif len(x[0]) == 3:
        axs[0].set_title("Parametry lotu")
        axs[0].set_ylabel("Kąt natarcia")
        axs[1].set_ylabel("Kąt nachylenia")
        axs[2].set_ylabel("Współczynnik nachylenia")
        axs[-1].set_xlabel("Czas [s]")
        process = 2
    else:
        axs[0].set_title("Stężenie komórek")
        axs[0].set_ylabel("Zdrowe CD4+")
        axs[1].set_ylabel("Zainfekowane CD4+")
        axs[2].set_ylabel("Prekursory LTC")
        axs[3].set_ylabel("Pomocniczo-niezależne LTC")
        axs[4].set_ylabel("Pomocniczo-zależne LTC")
        axs[-1].set_xlabel("Czas [dni]")
        process = 3
    kmax = len(y_zad)

    for i in range(len(x[0])):
        axs[i].step(range(kmax), y_zad[:, i], label=f"$x_{i+1}^{{zad}}$", color=f"C{i*2}")
        axs[i].step(range(kmax), x[0:kmax, i], label=f"$x_{i+1}$", color=f"C{i*2+1}")
        axs[i].legend()
        axs[i].grid()
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f"imgs/MPC_NO proces-{process} w={w}.png", dpi=300, bbox_inches='tight')
    
    plt.step(range(kmax-1), u[0:kmax-1])
    if process in [1, 3]:
        plt.xlabel("Czas [dni]")
    else:
        plt.xlabel("Czas [s]")
    plt.ylabel("Sterowanie")
    plt.grid()
    plt.title("Sterowanie")

    if save:
        plt.savefig(f"imgs/MPC_NO proces-{process} w={w} ster.png", dpi=300, bbox_inches='tight')
    plt.show()
