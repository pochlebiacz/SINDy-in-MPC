import numpy as np
import matplotlib.pyplot as plt
from pysindy.feature_library import PolynomialLibrary, FourierLibrary
import pysindy as ps
from generate_data import *


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

    return model


def get_tracking_model(Ts=0.1, thr=1e-5, deg=3):
    t = np.linspace(0, 20, int(20/Ts)+1)

    x0s = np.random.uniform(-0.1, 0.1, (500, 3))

    xss = []
    us = []

    for x0 in x0s:
        u = lambda t: x0[0] / 100. * np.sin(t)
        x = generate_discrete_tracking_data(t=t, x0=x0, T=Ts, u=u(t))
        xss.append(x)
        us.append(u(t))

    model = ps.SINDy(
        feature_library=PolynomialLibrary(degree=deg),
        optimizer=ps.STLSQ(threshold=thr),
        feature_names=[f'x{i+1}' for i in range(len(x0))]+['u'],
        discrete_time=True)
    
    model.fit(x=xss, u=us, multiple_trajectories=True)
    
    return model


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
        if w[i] > 0:
            axs[i].step(range(kmax), y_zad[:, i], label=f"$x_{i+1}^{{zad}}$", color=f"C{i*2}")
            axs[i].step(range(kmax), x[0:kmax, i], label=f"$x_{i+1}$", color=f"C{i*2+1}", linestyle='--')
        else:
            axs[i].step(range(kmax), x[0:kmax, i], label=f"$x_{i+1}$", color=f"C{i*2+1}")
        axs[i].legend()
        axs[i].grid()
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f"experiments/MPC_NO proces-{process} w={w}.pdf", bbox_inches='tight')
    
    plt.step(range(kmax-1), u[0:kmax-1])
    if process in [1, 3]:
        plt.xlabel("Czas [dni]")
    else:
        plt.xlabel("Czas [s]")
    plt.ylabel("Sterowanie")
    plt.grid()
    plt.title("Sterowanie")

    if save:
        plt.savefig(f"experiments/MPC_NO proces-{process} w={w} ster.pdf", bbox_inches='tight')
    plt.show()


def initialize(process, N, Nu, Ts, est='good'):
    if process == 1:
        x0 = [50, 20]
        y_zad = get_reference_population(x0)
        fun = population
        u = 5 * np.ones(len(y_zad)+Nu+N)
        bds = 100
        model = get_population_model(Ts, thr=1e-4, deg=2)
        if est == 'bad':
            model.coefficients()[1][3] += 0.1
        elif est == 'mid':
            model.coefficients()[1][3] += 0.05
    elif process == 2:
        x0 = [0, 0, 0]
        y_zad = get_reference_tracking(x0)
        fun = tracking
        u = np.zeros(len(y_zad)+Nu+N)
        bds = 0.002
        model = get_tracking_model(Ts, thr=1e-9, deg=3)
        if est == 'bad':
            model.coefficients()[0][3] += 0.05
        elif est == 'mid':
            model.coefficients()[0][3] += 0.025
    else:
        x05_stable = 0.05
        x0 = [11/20, 10/3, x05_stable, 0.0835-x05_stable, x05_stable]
        y_zad = get_reference_hiv(x0)
        fun = hiv
        u = 533/1078 * np.ones(len(y_zad)+Nu+N)
        bds = 1
        model = get_hiv_model(Ts, thr=1e-4, deg=3)
        if est == 'bad':
            model.coefficients()[1][6] += 0.02
        elif est == 'mid':
            model.coefficients()[1][6] += 0.01

    x = np.zeros((len(y_zad)+N, len(x0)))
    x[0:5, :] = x0

    return y_zad, fun, x, u, x.copy(), u.copy(), bds, model

    # 1. Stable initial condition: x1 = 5*x2-10*u, np. x, u = [(50, 20), 5]

def A_sum(A, n):
    result = np.zeros(A.shape)
    for i in range(n+1):
        result += np.linalg.matrix_power(A, i)
    return result