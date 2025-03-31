import numpy as np
from scipy.integrate import solve_ivp
from decimal import Decimal
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


def get_tables(libs, t, x, x_val, x0, x0_val, x_dot, u, u_val, T=1, discrete_time=False):
    for x_num in range(len(x0)):
        q = 'Biblioteka funkcji & Próg & $\Dot{x}'
        print(f'{q}_{x_num+1}$ & $E_{x_num+1}$ \\\\')
        for i, library in enumerate(libs):
            print(f'\\hline')
            for threshold in range(3):
                threshold = 10**(threshold-10) if i >= 2 else 100*10**(threshold-10)
                threshold = T * threshold if discrete_time else threshold
                name = ['Trygonometryczna (st. 2)', 'Trygonometryczna (st. 1)', 'Wielomiany (st. 3)', 'Wielomiany (st. 2)', 'Liniowa']
                model = ps.SINDy(
                    feature_library=library,
                    optimizer=ps.STLSQ(threshold=threshold),
                    feature_names=[f'x{i+1}' for i in range(len(x0))]+['u'],
                    discrete_time=discrete_time
                    )
                if discrete_time:
                    model.fit(x=x, u=u(t))
                else:
                    model.fit(x=x, x_dot=x_dot, u=u(t))
                try:
                    if discrete_time:
                        x_sim = model.simulate(x0=x0_val, t=len(t)+1, u=u_val(t))
                    else:
                        x_sim = model.simulate(x0=x0_val, t=t, u=u_val)
                    mse = ((x_sim - x_val)**2).mean(axis=0)
                    E = '%.3E' % Decimal(str(mse[x_num]))
                except:
                    E = '\infty'
                coeffs = ' + '.join(['%.3E' % Decimal(str(coeff))+' '+model.get_feature_names()[i] for i, coeff in enumerate(model.coefficients()[x_num]) if abs(model.coefficients()[x_num][i]) > threshold])
                if len(coeffs.split(' + ')) > 2:
                    eq = (coeffs.split(' + ')[0] + ' + ' + coeffs.split(' + ')[1] + '\dots').replace(' 1 +', ' +')
                else:
                    eq = coeffs
                if len(coeffs) == 0:
                    eq = '0,000'
                eq = eq.replace(' 1 +', ' +').replace('.', ',').replace('+ -', '- ').replace('sin', '\sin').replace('cos', '\cos').replace('(1 x1)', '(x_1)').replace('(1 x2)', '(x_2)').replace('x2', 'x_2').replace('(1 x3)', '(x_3)').replace('x3', 'x_3').replace('x1', 'x_1').replace('(1 u)', '(u)')
                thr = '%.0E' % Decimal(str(threshold))
                for pow in range(1, 10):
                    eq = eq.replace('E+00', '')
                    E = E.replace('E+00', '')
                    eq = eq.replace(f'E+0{pow}', f'\cdot 10^{pow}').replace(f'E-0{pow}', '\cdot 10^{'+f'{-pow}'+'}')
                    E = E.replace(f'E+0{pow}', f'\cdot 10^{pow}').replace(f'E-0{pow}', '\cdot 10^{'+f'{-pow}'+'}')
                    thr = thr.replace(f'1E+0{pow}', f'10^{pow}').replace(f'1E-0{pow}', '10^{'+f'{-pow}'+'}')
                print(f"{name[i]} & ${thr}$ & ${eq}$ & ${E.replace('.', ',')}$ \\\\")
        print('\n\n')


def get_reference_population(x0):
    x = np.zeros((100, 2))

    x[0:5, 0] = x0[0]
    x[5:20, 0] = 60
    x[20:60, 0] = 40
    x[60:80, 0] = 20
    x[80:100, 0] = 50

    x[0:5, 1] = x0[1]
    x[5:30, 1] = 40
    x[30:50, 1] = 50
    x[50:100, 1] = 20
    x[80:100, 1] = 35

    return x


def get_population_model(x0, Ts=0.1, thr=1e-4, deg=2):
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