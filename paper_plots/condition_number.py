''' 
Condition number plot.
'''


import sys
sys.path.append('../src/')
import math

import sympy as sym
from new_driver import GP_recipe1D, GP_recipe2D
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
def condition_number():


    #hyperparameters...
    #ell = 0.8
    xlim = (0,100)
    r_gps = [1,2,3]
    #Nxs = np.array([10, 25, 50, 75, 100, 150, 250, 500, 600, 700, 800])
    Nxs = np.linspace(10,2000,100,dtype=int)

    kernels = [kern.SE, kern.AS2, kern.DAS_V9]
    kernel_names = ["SE", "NN", "DAS"]

    #kernels will all be the same color
    colors = ['b', '#18BF83', 'r']

    #each r_gp will have same linestyle
    linestyle = ["-", "--", ":"]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
    plt.subplots_adjust(wspace=0.3)  # Increase horizontal spacing between subplots

    # Set larger font sizes for tick labels
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    for i in range(len(kernels)):


        for k in range(len(r_gps)):
            
            conditionNumbers = []


            for j in range(len(Nxs)):

                g = Grid1D(xlim, Nxs[j], r_gps[k])
                g.fill_grid(np.sin)
                dx = g.x[1] - g.x[0]
                ell = 1.8


                gprecipe= GP_recipe1D(g, r_gps[k], ell=ell, stencil_method ="center", high_precision=True)

                x_predict = g.x_int + dx/2

                K, K_star = gprecipe.convert_custom(x_predict,kernels[i], kernels[i], hack_for_getting_K=True)

                #print(type(K))

                conditionNumber = mm.cond(K)
                conditionNumbers.append(conditionNumber)
        
            ax1.loglog(100/Nxs, conditionNumbers, linestyle[k], color=colors[i], label=kernel_names[i] + r" $r_{gp}$ = " + str(r_gps[k]))

    ax1.set_xlabel("$\mathbf{\Delta x}$", fontsize=16)
    ax1.set_ylabel("Condition Number", fontsize=16)
    ax1.legend(loc='upper right', fontsize=12)
    #plt.show()
    #plt.savefig("condition_number.png")

        





    #N



    #hyperparameters...
    #ell = 0.8
    xlim = (0,100)
    r_gp = 2

    Nxs = np.array([10, 25, 50, 75, 100, 150, 250, 500, 1000, 10000, 20000])
    #Nxs = np.linspace(10,1000,100,dtype=int)

    sigmas = [1, 10, 100]
    ells = [1,10,100]

    kernels = [kern.SE, kern.AS2, kern.DAS_V9]
    kernel_names = ["SE", "NN", "DAS"]
    #kernels will all be the same color
    colors = ['b', '#18BF83', 'r']

    #each r_gp will have same linestyle
    linestyle = ["-", "--", ":"]






    for i in range(len(kernels)):


        for k in range(len(sigmas)):
            
            conditionNumbers = []


            for j in range(len(Nxs)):

                g = Grid1D(xlim, Nxs[j], r_gp)
                g.fill_grid(np.sin)
                dx = g.x[1] - g.x[0]
                ell = 6*dx

                gprecipe= GP_recipe1D(g, r_gp, ell=ells[k], sigma=sigmas[k] if i != 2 else 1,  stencil_method ="center", high_precision=True)

                x_predict = g.x_int + dx/2

                K, K_star = gprecipe.convert_custom(x_predict,kernels[i], kernels[i], hack_for_getting_K=True)

                conditionNumber = mm.cond(K)
                conditionNumbers.append(conditionNumber)


            

            if i == 0:
                exponent = int(math.log10(ells[k]))
                superscript = str(exponent).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
                ax2.loglog(100/Nxs, conditionNumbers, linestyle[k], color=colors[i], label=kernel_names[i] + r" $\ell$ = 10" + superscript)
            elif i == 1:
                exponent = int(math.log10(sigmas[k]))
                superscript = str(exponent).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
                ax2.loglog(100/Nxs, conditionNumbers, linestyle[k], color=colors[i], label=kernel_names[i] + r" $\sigma$ = 10" + superscript)
            else:
                # For DAS, just use a fixed sigma=1 and don't show it in the label
                if k == 0:  # Only plot once for DAS, with the first linestyle
                    ax2.loglog(100/Nxs, conditionNumbers, linestyle[k], color=colors[i], label=kernel_names[i])
    ax2.set_xlabel("$\mathbf{\Delta x}$", fontsize=16)
    ax2.set_ylabel("Condition Number", fontsize=16)
    ax2.legend(loc='upper right', fontsize=12)



    # plt.show()
    plt.savefig("condition_number.png", dpi=400)

    plt.show()

if __name__ == "__main__":
    condition_number()