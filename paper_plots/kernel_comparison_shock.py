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

plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
#hyperparameters...

def kernel_comparison_shock():

    #plt.style.use('seaborn-v0_8-paper')  

    #plot
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    #fig.suptitle('Shock Capturing Kernel Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #hyperparameters.
    kernels = [kern.SE, kern.AS2, kern.DAS_V9]
    Nxs = [100,200,400]
    sigmas = [100, 10, 1]
    ells = [10, 1, .1]
    xlim = (-2,2)
    r_gps = [1,2,3]
    marker = ["+-", "o--", "^-", "*-"]
    colors = ['b', '#18BF83', 'r']


    #shock
    def f(x):
        y = np.zeros_like(x)

        for i in range(len(x)):
            
            if x[i] < 0:
                y[i] = 0
            else:
                y[i] = 1
        return y

    bounds = axs[0, 1].get_position().bounds

    # Calculate inset position based on the subplot's bounds
    # # The inset is placed in the bottom right of the subplot with a slight offset
    # inset_width = bounds[2] * 0.3  # 30% of the subplot's width
    # inset_height = bounds[3] * 0.3  # 30% of the subplot's height
    # inset_left = bounds[0] + bounds[2] * 0.7  # 70% from the left of the subplot
    # inset_bottom = bounds[1] + bounds[3] * 0.2  # 20% from the bottom of the subplot

    # # Define the inset position
    # inset_position = [inset_left, inset_bottom, inset_width, inset_height]
    # ax_inset = fig.add_axes(inset_position)


    for i,kernel in enumerate(kernels):
        g = Grid1D(xlim, 200, 2)
        g_exact = Grid1D(xlim, 1000, 2)


        for j,Nx in enumerate(Nxs):
            r_gp = 2
            #g = Grid1D(xlim, Nx, r_gp)
            g.fill_grid(f)
            dx = g.x[1] - g.x[0]

            ell = dx*12

            gprecipe = GP_recipe1D(g, r_gp, ell=ells[j], sigma=sigmas[j], stencil_method ="center", high_precision=True)

            x_predict = g.x_int + dx/2
            x_predict = x_predict[:-1]

            y_predict = gprecipe.convert_custom(x_predict,kernel, kernel)
            if i == 0:
                axs[i,0].plot(x_predict, y_predict, marker[j], c=colors[j], fillstyle='none', label="$\ell$ = " + str(ells[j]), markersize=7)
            elif i ==1:
                axs[i,0].plot(x_predict, y_predict, marker[j], c=colors[j], fillstyle='none', label="$\sigma$ = = " + str(sigmas[j]), markersize=7)
            elif i == 2:
                if j == 2:
                    axs[i,0].plot(x_predict, y_predict, marker[j], c=colors[j], fillstyle='none', label="DAS")

        for j,r_gp in enumerate(r_gps):
            Nx = 400
            g = Grid1D(xlim, Nx, r_gp)
            g.fill_grid(f)
            dx = g.x[1] - g.x[0]

            ell = .1
            sigma = 100


            gprecipe = GP_recipe1D(g, r_gp, ell=ell, sigma=sigma, stencil_method ="center", high_precision=True)
            gprecipe_das = GP_recipe1D(g, r_gp, ell=ell, sigma=1, stencil_method ="center", high_precision=True)

            x_predict = g.x_int + dx/2
            x_predict = x_predict[:-1]

            y_predict = gprecipe.convert_custom(x_predict,kernel, kernel)

            if i == 2:
                y_predict = gprecipe_das.convert_custom(x_predict,kernel, kernel)
            axs[i,1].plot(x_predict, y_predict, marker[j], c=colors[j], fillstyle='none', label="Prediction r_gp = " + str(r_gp), markersize=7)

            #inset.
            # if i==0:
            #     ax_inset.plot(x_predict, y_predict, marker[j], c=colors[j], fillstyle='none', label="Prediction r_gp = " + str(r_gp), markersize=4)
            #     ax_inset.set_ylim((-4.2, 4.2))
            #     ax_inset.set_xlim((-.25, .25))

        
        axs[i,0].plot(g_exact.x_int, f(g_exact.x_int), c='black', label="Exact Solution")
        axs[i,1].plot(g_exact.x_int, f(g_exact.x_int), c='black', label="Exact Solution")
        # if i==0:
        #     ax_inset.plot(g_exact.x_int, f(g_exact.x_int), c='black', label="Exact Solution")


        
    
        if i == 0:
            axs[i,0].set_title("SE Kernel Hyperparameter Comparison")
            axs[i,1].set_title(r"SE Kernel Grid $r_{gp}$ Comparison")
        elif i == 1:
            axs[i,0].set_title("NN Kernel Hyperparameter Comparison")
            axs[i,1].set_title(r"NN Kernel Grid $r_{gp}$ Comparison")
        elif i == 2:
            axs[i,0].set_title("DAS Kernel")
            axs[i,1].set_title(r"DAS Kernel Grid $r_{gp}$ Comparison")

        for j in [0,1]:
            axs[i,j].set_xlabel("$\mathbf{x}$")
            axs[i,j].set_ylabel("$\mathbf{y}$")
            axs[i,j].legend(loc="center right")
            axs[i,j].set_xlim((-.2,.2))
            axs[i,j].set_ylim((-0.2, 1.3))


    
    plt.tight_layout()
    plt.savefig("kernel_shock.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    kernel_comparison_shock()