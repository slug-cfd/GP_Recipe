''' 
Compund waves.
'''

import sys
sys.path.append('../src/')

import sympy as sym
from driver import GP_driver_data
from new_driver import GP_recipe1D, GP_recipe2D
import inspect
import numpy as np
import matplotlib.pyplot as plt
from grids import Grid1D, Grid2D

import kernels as kern
import mpmath as mm
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def compound_waves():
    plt.style.use('seaborn-v0_8-bright')  


    #hyperparameters...
    xlims = [(0,1), (0, 1.5), (-4,4)]
    r_gp = 2
    fig, (ax2, ax3) = plt.subplots(2,1, figsize=(8,4))
    Nx = 100
    NxSolution = 1000




    def f2(x):
        """
        f1(x) = 0 if 0 <= x < 0.3,
        f1(x) = e^(-2x) * sin(10πx) + 1/2 if 0.3 <= x <= 0.6,
        f1(x) = 1 if 0.6 < x <= 1.0.
        """
        result = np.zeros_like(x)
        condition1 = (0 <= x) & (x < 0.3)
        condition2 = (0.3 <= x) & (x <= 0.6)
        condition3 = (0.6 < x) & (x <= 1.0)

        result[condition2] = np.exp(-2*x[condition2]) * np.sin(10*np.pi*x[condition2]) + 0.5
        result[condition3] = 1

        return result

    def f3(x):
        """
        f2(x) = sin(πx) if |x| >= 1.0,
        f2(x) = 3 if -1 < x <= -0.5,
        f2(x) = 1 if -0.5 < x <= 0,
        f2(x) = 3 if 0 < x <= 0.5,
        f2(x) = 2 if 0.5 < x <= 1.0.
        """
        result = np.where(np.abs(x) >= 1.0, np.sin(np.pi*x), 0)
        result = np.where((-1 < x) & (x <= -0.5), 3, result)
        result = np.where((-0.5 < x) & (x <= 0), 1, result)
        result = np.where((0 < x) & (x <= 0.5), 3, result)
        result = np.where((0.5 < x) & (x <= 1.0), 2, result)

        return result



    def f1(x):
        y = np.zeros_like(x)

        for i in range(len(x)):
            
            if x[i] < 0.5:
                y[i] = 0
            else:
                y[i] = 1
        return y



    g1 = Grid1D(xlims[0], Nx, r_gp)
    g2 = Grid1D(xlims[1], Nx, r_gp)
    g3 = Grid1D(xlims[2], Nx, r_gp)
    #referenceGrid = Grid1D(xlims[2], NxSolution, r_gp)


    g1.fill_grid(f1)
    g2.fill_grid(f2)
    g3.fill_grid(f3)


    dx1 = g1.x[1] - g1.x[0]
    dx2 = g2.x[1] - g2.x[0]
    dx3 = g3.x[1] - g3.x[0]

    x_predict1 = g1.x_int + dx1/2
    x_predict2 = g2.x_int + dx2/2
    x_predict3 = g3.x_int + dx3/2

    x_predict1 = x_predict1[:-1]
    x_predict2 = x_predict2[:-1]
    x_predict3 = x_predict3[:-1]

    kernels = [kern.AS2, kern.SE, kern.DAS_V9]

    for i,kernel in enumerate(kernels):



        gprecipe1 = GP_recipe1D(g1, r_gp, ell=12*dx1, stencil_method ="center", high_precision=True)
        gprecipe2 = GP_recipe1D(g2, r_gp, ell=12*dx2, stencil_method ="center", high_precision=True)
        gprecipe3 = GP_recipe1D(g3, r_gp, ell=12*dx3, stencil_method ="center", high_precision=True)


        y_predict1 = gprecipe1.convert_custom(x_predict1,kernel, kernel)
        y_predict2 = gprecipe2.convert_custom(x_predict2,kernel, kernel)
        y_predict3 = gprecipe3.convert_custom(x_predict3,kernel, kernel)

        linewidth = 0.5
        markersize = 4

        if i == 0:
            #ax1.plot(x_predict1, y_predict1, '+-', label="NN Kernel", markersize=markersize, linewidth=linewidth)
            ax2.plot(x_predict2, y_predict2, '+-', label="NN Kernel", markersize=markersize, linewidth=linewidth)
            ax3.plot(x_predict3, y_predict3, '+-', label="NN Kernel", markersize=markersize, linewidth=linewidth)
        if i == 1:
            seColor = "limegreen"
            #seColor = 'green'
            #ax1.plot(x_predict1, y_predict1, 'o--', c=seColor, fillstyle='none', label="SE Kernel", markersize=markersize, linewidth=linewidth)
            ax2.plot(x_predict2, y_predict2, 'o--', c=seColor, fillstyle='none', label="SE Kernel", markersize=markersize, linewidth=linewidth)
            ax3.plot(x_predict3, y_predict3, 'o--', c=seColor, fillstyle='none', label="SE Kernel", markersize=markersize, linewidth=linewidth)

        elif i == 2:
            #ax1.plot(x_predict1, y_predict1, '^-', c='r', fillstyle='none', label="DAS Kernel", markersize=markersize-1, linewidth=linewidth)
            ax2.plot(x_predict2, y_predict2, '^-', c='r', fillstyle='none', label="DAS Kernel", markersize=markersize-1, linewidth=linewidth)
            ax3.plot(x_predict3, y_predict3, '^-', c='r', fillstyle='none', label="DAS Kernel", markersize=markersize-1, linewidth=linewidth)





    x_UF1 = np.linspace(xlims[0][0], xlims[0][1], NxSolution)
    # ax1.plot(x_UF1,f1(x_UF1), label="Exact Solution", c='k', linewidth=.5)
    #ax1.set_xlim(xlims[0])

    x_UF2 = np.linspace(xlims[1][0], xlims[1][1], NxSolution)
    ax2.plot(x_UF2,f2(x_UF2), label="Exact Solution", c='k', linewidth=.5)
    #ax2.set_xlim(xlims[1])

    x_UF3 = np.linspace(xlims[2][0], xlims[2][1], NxSolution)
    ax3.plot(x_UF3,f3(x_UF3),label='Exact Solution', c='k', linewidth=.5)
    #ax3.set_xlim(xlims[2])





    # ax1.legend()
    ax2.legend()
   # ax3.legend()

    # ax1.set_ylim(-.2, 1.2)
    ax2.set_ylim(-.2, 1.2)
    
    # Add mathematical script axis labels
    # ax1.set_xlabel("$x$")
    # ax1.set_ylabel("$f(x)$")
    ax2.set_xlabel("$\mathbf{x}$")
    ax2.set_ylabel("$\mathbf{f_{1}(x)}$")
    ax3.set_xlabel("$\mathbf{x}$")
    ax3.set_ylabel("$\mathbf{f_{2}(x)}$")

    plt.tight_layout()

    plt.savefig("compound_waves.png", dpi=400)

    plt.show()

if __name__ == "__main__":
    compound_waves()