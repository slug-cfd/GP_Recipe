
import sys
sys.path.append('../../')

import sympy as sym
from driver import GP_driver_data
from new_driver import GP_recipe1D, GP_recipe2D
import inspect
import numpy as np
import matplotlib.pyplot as plt
from grids import Grid1D, Grid2D

import kernels as kern
import mpmath as mm



def sin_derivatives():

    #hyperparameters...
    xlim = (0, 2*np.pi)
    r_gp = 2
    Nx = 50

    def f(x):
        return np.sin(x)

    g = Grid1D(xlim, Nx, r_gp)
    g.fill_grid(f)
    dx = g.x[1] - g.x[0]

    ell = 12*dx


    gprecipe = GP_recipe1D(g, r_gp, ell=ell, stencil_method ="center", high_precision=False)


    x_predict = g.x_int + dx/2
    x_predict = x_predict[:-1]

    plt.plot(g.x_int, g.internal_grid, '.', label="Input Data", c='k')

    #1st derivative
    y_predict = gprecipe.convert_custom(x_predict,kern.SE, sym.diff(kern.SE, kern.y_sym))
    plt.plot(x_predict, y_predict, '.', label="1st Derivative Prediction", c='blue')
    plt.plot(x_predict, np.cos(x_predict), label="1st Derivative Solution", alpha=0.5, c='blue')

    #2nd derivative
    y_predict = gprecipe.convert_custom(x_predict,kern.SE, sym.diff(kern.SE, kern.y_sym, 2))
    plt.plot(x_predict, y_predict, '.', label="2nd Derivative Prediction", c='red')
    plt.plot(x_predict, -np.sin(x_predict), label="2nd Derivative Solution", alpha=0.5, c='red')


    #3rd derivative
    y_predict = gprecipe.convert_custom(x_predict,kern.SE, sym.diff(kern.SE, kern.y_sym, 3))
    plt.plot(x_predict, y_predict, '.', label="3rd Derivative Prediction", c='green')
    plt.plot(x_predict, -np.cos(x_predict), label="3rd Derivative Solution", alpha=0.5, c='green')



    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sin(x) input data computing derivatives 1-3")
    plt.legend()
    plt.savefig("sin_derivatives", dpi=400)