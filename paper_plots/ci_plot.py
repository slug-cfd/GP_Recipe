import sys
sys.path.append('../src/')

import sympy as sym
from driver import GP_driver_data
from new_driver import GP_recipe1D, GP_recipe2D
import inspect
import numpy as np
import matplotlib.pyplot as plt
from grids import Grid1D, Grid2D
import scipy.stats as stats
import matplotlib.colors as mcolors

import kernels as kern
import mpmath as mm


def ci_plot_new():
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2(np.linspace(0, 1, 8)))
    
    # Increase font sizes for better readability
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    hex_color = '#332288'
    rgb_color = mcolors.hex2color(hex_color)

    kernel = kern.SE
    #kernel = kern.AS2

    #hyperparameters...
    xlim = (0, 13)
    r_gp = 2
    Nx = 25
    def f(x):
        y = np.zeros_like(x)

        for i in range(len(x)):
            if x[i] < 10:
                y[i] = np.sin(x[i])*x[i]
            elif x[i] < 11:
                y[i] = -6
            else:
                y[i] = 10
        return y
    g = Grid1D(xlim, Nx, r_gp)
    g.fill_grid(f)
    dx = g.x[1] - g.x[0]
    ell = 0.8

    #Solution Prediction at cell interfaces
    gprecipe = GP_recipe1D(g, r_gp, ell=ell, stencil_method ="center", high_precision=True, precision=113, mean_method="zero")
    x_predict = g.x_int + dx/2
    x_predict = x_predict[:-1]
    y_predict, sigmas = gprecipe.convert_custom_ci(x_predict, kernel, kernel, kernel)

    # Convert mpmath objects to float
    for i in range(len(sigmas)):
        sigmas[i] = float(mm.sqrt(mm.sqrt((sigmas[i]*sigmas[i])))) # mpmath has no abs

    # Use 3-sigma error bars
    error_bars = 20 * np.array(sigmas).astype(float)

    plt.figure(figsize=(10,6))
    
    # High resolution underlying function
    x_high_res = np.linspace(xlim[0], xlim[1], 200)
    y_high_res = f(x_high_res)
    
    plt.plot(x_high_res, y_high_res, label="Underlying Function", color='black', linewidth=2)
    plt.plot(g.x_int, g.internal_grid, 'o', label="Training Data", color='blue', markersize=6)
    
    # Plot predictions with error bars - same color for points and error bars
    plt.errorbar(x_predict, y_predict, yerr=error_bars, fmt='s', 
                 label="Prediction with 20σ error bars", capsize=3, 
                 markersize=5, color='red', ecolor='red', alpha=0.8)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("ci_plot.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    ci_plot_new() 