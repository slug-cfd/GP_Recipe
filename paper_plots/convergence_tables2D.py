'''
Generating all the tables for converegence.

Grid resolution by:
L1 Error
EOC
L2 Error
EOC

Then separate table for r_gp…

Convergence plot pt to pt. Different r_gp...
'''

import sys
sys.path.append('../')
import sympy as sym
from driver import GP_driver_data
from new_driver import GP_recipe1D, GP_recipe2D
import inspect
import numpy as np
import matplotlib.pyplot as plt
from grids import Grid1D, Grid2D
import kernels as kern
import time
import error
import mpmath as mm
mm.mp.dps = 1000
start_time = time.time()

PI = 3.14159

r_gps = [1,2,3]
Nxs = np.array([16,32,64,128,256])
# r_gps = [1,2]
# Nxs = np.array([25,50, 100])
high_precision = True
precision_num = 113


# Replace with a more complex function that's still analytically integrable
f = lambda X, Y: np.exp(-2*X) * np.sin(4*np.pi*Y) + X * np.cos(2*np.pi*X) * np.exp(-Y)

#hard code laplacian that goes along with the plot we're showing of it.
#this is the data function.
f_lap =  lambda X,Y : np.sin(X) * np.sin(Y)
#exact solution of lap
L_lap_np = lambda X,Y: -2 * np.sin(X) * np.sin(Y)
L_lap = sym.diff(kern.SE_2d, kern.x_sym1, kern.x_sym1) + sym.diff(kern.SE_2d, kern.y_sym1, kern.y_sym1)


# Convert f to a sympy expression
X_sym, Y_sym = sym.symbols('X Y')
f_sympy = sym.exp(-2*X_sym) * sym.sin(4*sym.pi*Y_sym) + X_sym * sym.cos(2*sym.pi*X_sym) * sym.exp(-Y_sym)
f_lap_sympy = sym.sin(X_sym) * sym.sin(Y_sym)

# Calculate derivatives symbolically
f_dx_sympy = sym.diff(f_sympy, X_sym)  # d/dx
f_dy_sympy = sym.diff(f_sympy, Y_sym)  # d/dy
f_dxx_sympy = sym.diff(f_sympy, X_sym, 2)  # d^2f/dX^2
f_dyy_sympy = sym.diff(f_sympy, Y_sym, 2)  # d^2f/dY^2
f_dxdy_sympy = sym.diff(sym.diff(f_sympy, X_sym), Y_sym)  # d/dx d/dy
#f_laplacian_sympy = f_dxx_sympy + f_dyy_sympy  # d^2f/dX^2 + d^2f/dY^2
f_laplacian_sympy = sym.diff(f_lap_sympy, X_sym, 2) + sym.diff(f_lap_sympy, Y_sym, 2)

# Convert sympy expressions to numpy functions
f_dx_np = sym.lambdify((X_sym, Y_sym), f_dx_sympy, 'numpy')
f_dy_np = sym.lambdify((X_sym, Y_sym), f_dy_sympy, 'numpy')
f_dxx_np = sym.lambdify((X_sym, Y_sym), f_dxx_sympy, 'numpy')
f_dyy_np = sym.lambdify((X_sym, Y_sym), f_dyy_sympy, 'numpy')
f_dxdy_np = sym.lambdify((X_sym, Y_sym), f_dxdy_sympy, 'numpy')
f_laplacian_np = sym.lambdify((X_sym, Y_sym), f_laplacian_sympy, 'numpy')


f_integral_sympy_x = sym.integrate(f_sympy, X_sym)
f_integral_sympy_xy = sym.integrate(f_integral_sympy_x, Y_sym)

# Convert sympy expression for the indefinite integral to a numpy function
# f_integral_np = sym.lambdify((X_sym, Y_sym), f_integral_sympy_xy, 'numpy') # This line causes issues and is not needed for the numerical approach

# Updated implementation of exact_volume_average_f_grid for the more complex function
def exact_volume_average_f_grid(X_cell_centers, Y_cell_centers, dx_val, dy_val):
    """
    Computes the exact volume average of 
    exp(-2x)*sin(4πy) + x*cos(2πx)*exp(-y) over grid cells.
    
    Args:
        X_cell_centers: Numpy array of X-coordinates of cell centers.
        Y_cell_centers: Numpy array of Y-coordinates of cell centers.
        dx_val: Width of the cells.
        dy_val: Height of the cells.
        
    Returns:
        Numpy array of volume-averaged values, same shape as X_cell_centers.
    """
    X_cc_np = np.asarray(X_cell_centers)
    Y_cc_np = np.asarray(Y_cell_centers)
    original_shape = X_cc_np.shape
    
    # For high precision calculations
    if high_precision:
        # Create mpmath versions of constants
        four_pi = mm.mpf('4.0') * mm.pi
        two_pi = mm.mpf('2.0') * mm.pi
        
        # Convert to flat arrays for processing
        flat_X = X_cc_np.flatten()
        flat_Y = Y_cc_np.flatten()
        results = np.empty_like(flat_X, dtype=object)
        
        for i, (x_c, y_c) in enumerate(zip(flat_X, flat_Y)):
            x_m = mm.mpf(str(float(x_c) - float(dx_val)/2))
            x_p = mm.mpf(str(float(x_c) + float(dx_val)/2))
            y_m = mm.mpf(str(float(y_c) - float(dy_val)/2))
            y_p = mm.mpf(str(float(y_c) + float(dy_val)/2))
            
            # Term 1: exp(-2x)*sin(4πy)
            x_term1 = (mm.exp(-2*x_m) - mm.exp(-2*x_p)) / -2  # Integral of exp(-2x)
            y_term1 = (mm.cos(four_pi * y_m) - mm.cos(four_pi * y_p)) / -four_pi  # Integral of sin(4πy)
            term1 = x_term1 * y_term1
            
            # Term 2: x*cos(2πx)*exp(-y)
            # For ∫x*cos(2πx)dx use integration by parts:
            # ∫x*cos(2πx)dx = x*sin(2πx)/(2π) - ∫sin(2πx)/(2π)dx
            #                = x*sin(2πx)/(2π) + cos(2πx)/(4π²)
            x_term2 = ((x_p*mm.sin(two_pi*x_p) - x_m*mm.sin(two_pi*x_m))/two_pi) + \
                      ((mm.cos(two_pi*x_p) - mm.cos(two_pi*x_m))/(4*mm.pi*mm.pi))
            
            # For ∫exp(-y)dy = -exp(-y)
            y_term2 = (mm.exp(-y_m) - mm.exp(-y_p))
            term2 = x_term2 * y_term2
            
            # Combine terms and normalize by cell area
            cell_area = mm.mpf(str(float(dx_val))) * mm.mpf(str(float(dy_val)))
            results[i] = (term1 + term2) / cell_area
        
        return results.reshape(original_shape)
    
    else:
        # Standard precision calculation
        four_pi = 4.0 * np.pi
        two_pi = 2.0 * np.pi
        
        # Handle both meshgrid and normal array inputs
        if len(X_cc_np.shape) > 1:
            # For meshgrid style inputs
            dx = np.ones_like(X_cc_np) * dx_val
            dy = np.ones_like(Y_cc_np) * dy_val
        else:
            # For 1D arrays
            dx = np.ones_like(X_cc_np) * dx_val
            dy = np.ones_like(Y_cc_np) * dy_val
        
        # Calculate half widths for cell boundaries
        dx_half = dx / 2
        dy_half = dy / 2
        
        # Calculate the analytical integral over each cell
        x_minus = X_cc_np - dx_half
        x_plus = X_cc_np + dx_half
        y_minus = Y_cc_np - dy_half
        y_plus = Y_cc_np + dy_half
        
        # Term 1: exp(-2x)*sin(4πy)
        x_term1 = (np.exp(-2*x_minus) - np.exp(-2*x_plus)) / -2
        y_term1 = (np.cos(four_pi * y_minus) - np.cos(four_pi * y_plus)) / -four_pi
        term1 = x_term1 * y_term1
        
        # Term 2: x*cos(2πx)*exp(-y)
        x_term2 = ((x_plus*np.sin(two_pi*x_plus) - x_minus*np.sin(two_pi*x_minus))/two_pi) + \
                   ((np.cos(two_pi*x_plus) - np.cos(two_pi*x_minus))/(4*np.pi*np.pi))
        y_term2 = (np.exp(-y_minus) - np.exp(-y_plus))
        term2 = x_term2 * y_term2
        
        # Final result is the sum of terms normalized by cell area
        result = (term1 + term2) / (dx * dy)
        
        return result


# Define other exact solutions based on the new f
exact_f = f # Renaming for clarity, f is already the point-wise exact solution
ddx = f_dx_np
ddy = f_dy_np
ddx2 = f_dxx_np
ddy2 = f_dyy_np
dxdy = f_dxdy_np
L_lap_np = f_laplacian_np 




exact_sols = [f, f, f, ddx, ddy, dxdy, exact_volume_average_f_grid, f, dxdy, ddx2, ddy2, ddx, ddy, ddx2, L_lap_np]
names = ["Pt_to_Pt_2D_xdx", "Pt_to_Pt_2D_ydx", "Pt_to_Pt_2D", "Pt_to_ddx_2D", "Pt_to_ddy_2D", "Pt_to_dxdy_2D", "Pt_to_VolAve_2D",
         "VoleAve_to_pt", "VoleAve_to_dxdy", "VoleAve_to_dxdx", "VoleAve_to_dydy", "VoleAve_to_dx", "VoleAve_to_dy", "Pt_to_dxdx", "Pt_to_Laplacian"]

x_stars = [(1/2, 0), (0, 1/2), (1/2, 1/2), (1/2, 1/2), (1/2, 1/2), (1/2, 1/2), (1/2, 1/2),(1/2, 1/2),(1/2, 1/2),(1/2, 1/2),(1/2, 1/2),(1/2, 1/2),(1/2, 1/2), (1/2,1/2), (1/2, 1/2)]


kint = sym.integrate(kern.SE_2d, kern.x_sym1)
left_integral = kint.subs(kern.x_sym1, kern.x_sym1-kern.dx_sym/2)
right_integral = kint.subs(kern.x_sym1, kern.x_sym1+kern.dx_sym/2)
Tx = (right_integral - left_integral)/kern.dx_sym

kint = sym.integrate(Tx, kern.y_sym1)
left_integral = kint.subs(kern.y_sym1, kern.y_sym1-kern.dx_sym/2)
right_integral = kint.subs(kern.y_sym1, kern.y_sym1+kern.dx_sym/2)
T_2D = (right_integral - left_integral)/kern.dx_sym
T_2D = T_2D.simplify()


k_kernels = [kern.SE_2d, kern.SE_2d, kern.SE_2d, kern.SE_2d, kern.SE_2d, kern.SE_2d, kern.SE_2d,  kern.C_2D, kern.C_2D, kern.C_2D, kern.C_2D, kern.C_2D, kern.C_2D, kern.SE_2d, kern.SE_2d]
k_star_kernels = [kern.SE_2d, kern.SE_2d, kern.SE_2d, sym.diff(kern.SE_2d, kern.x_sym1), sym.diff(kern.SE_2d, kern.y_sym1),
                  sym.diff(sym.diff(kern.SE_2d, kern.x_sym1), kern.y_sym1), T_2D,  T_2D, sym.diff(sym.diff(T_2D, kern.x_sym1), kern.y_sym1), sym.diff(sym.diff(T_2D, kern.x_sym1), kern.x_sym1), sym.diff(sym.diff(T_2D, kern.y_sym1), kern.y_sym1), sym.diff(T_2D, kern.x_sym1),sym.diff(T_2D, kern.y_sym1) , sym.diff(sym.diff(kern.SE_2d, kern.x_sym1), kern.x_sym1), L_lap]
                

assert(len(names) == len(exact_sols))
assert(len(names) == len(k_kernels))
assert(len(names) == len(k_star_kernels))
assert(len(names) == len(x_stars))

#for k in [2, 6,7]:  #2 is pt to pt, 6 is pt to volumeaverage, 7 is laplacian, #5 is pt to dxdy
#for k in [7,14,15]: #14 is VoleAve_to_pt #15 is vole average to dxdy. #16 is vole ave to dxdx #17 is vole ave to dydy #18 is vole ave to dx #19 is vole ave to dy

#for k in [3,4,5,16,15,17,18,19,20]:
# for k in [15,16,5,20]:

for k in range(14,len(names)):
    print("----", names[k])
    error_l1 = np.zeros((3,5))
    error_l2 = np.zeros((3,5))
    error_linf = np.zeros((3,5))
    EOC_m_L1 = np.zeros((3,5))
    EOC_m_L2 = np.zeros((3,5))
    EOC_m_Linfty = np.zeros((3,5))

    if high_precision:
        error_l1 = mm.matrix(error_l1.tolist())  
        error_l2 = mm.matrix(error_l2.tolist())  
        error_linf = mm.matrix(error_linf.tolist())  
   

    
    fig, (ax1) = plt.subplots(1, 1, sharey=True)

    for i, r_gp in enumerate(r_gps):
        for j, Nx in enumerate(Nxs):

            print(f"Running rgp {r_gp} Nxs: {Nx}")


            if k == 14:
                #special for laplcian:
                xlim = (0, 2*np.pi)
                g2d = Grid2D(xlim, xlim, Nx, Nx, r_gp)
                g2d.fill_grid(f_lap) 
                ell = 0.05 * 2*np.pi


            else:
                xlim = (0, 1)
                g2d = Grid2D(xlim, xlim, Nx, Nx, r_gp)

                g2d.fill_grid(exact_f) #to be overwritten, but we need to create the grid object...

                ell = 0.05

                if k_kernels[k] == kern.SE_2d:
                    #pointwise inputs
                    g2d.fill_grid(exact_f)

                elif k_kernels[k] == kern.C_2D:
                    #volume average inputs

                    # Define a wrapper function to pass to fill_grid
                    def volume_avg_f_for_g2d_fill(grid_X_centers, grid_Y_centers):
                        # g2d.dx and g2d.dy are the cell dimensions for the grid g2d
                        return exact_volume_average_f_grid(grid_X_centers, grid_Y_centers, g2d.dx, g2d.dy)
                    
                    g2d.fill_grid(volume_avg_f_for_g2d_fill)


                else:
                    print("Unsupported inputs type, please implemenet")
                    sys.exit()
            
              
            dx = (xlim[1] - xlim[0]) / Nx
            dy = dx


            #exception policy for r_gp = 1.
            # if r_gp == 1:
            #     stencil = "square"
            # else:
            #     stencil = "cross"
            stencil = "blocky_diamond"
            
            #ell = 0.4 works for up to 200 but fails for 400 grid points. 
            #for xlim - 0,2pi
            #ell = 0.3
            
            gp_recipe2d = GP_recipe2D(g2d, r_gp, ell=ell, stencil_method=stencil, high_precision=high_precision, precision=precision_num, mean_method="zero")
            # gp_recipe2d_diamond = GP_recipe2D(g2d, r_gp, ell=ell, stencil_method="diamond", high_precision=high_precision, mean_method="zero")


            X, Y = np.meshgrid(g2d.x, g2d.y)

            x_predict = g2d.x_int + g2d.dx * x_stars[k][0]
            x_predict = x_predict[:-1]
            y_predict = g2d.y_int+ g2d.dx * x_stars[k][1]
            y_predict = y_predict[:-1]

            X_predict, Y_predict = np.meshgrid(x_predict, y_predict)

            points_predict = [[x, y] for x, y in zip(X_predict.flatten(), Y_predict.flatten())]


  
            # for point in points_predict:
            #     #2D stencil test.
            #     X, Y = np.meshgrid(g2d.x, g2d.y)
            #     Z = X*0
            #     plt.scatter(X, Y, c='k', alpha=0.2)

            #     #point = points_predict[0]
            #     x,y,g = gp_recipe2d.get_stencil(point)
            #     x_diamond,y_diamond,g_diamond = gp_recipe2d_diamond.get_stencil(point)

            #     print("x: ", x)
            #     print("y: ", y)
            #     print("x_diamond: ", x_diamond)
            #     print("y_diamond: ", y_diamond)

            #     #plot 
            #     plt.scatter(x,y, c='red')
            #     plt.scatter(x_diamond,y_diamond, c='blue')
            #     plt.scatter(point[0],point[1], c='green') 
            #     plt.show()


            sol = gp_recipe2d.convert_custom(points_predict, k_kernels[k], k_star_kernels[k], stationary=True)
            #K_diamond,k_star_diamond = gp_recipe2d_diamond.convert_custom([points_predict[0]], k_kernels[k], k_star_kernels[k], stationary=True,hack_for_getting_K=True)


            # Calculate exact solution considering if it needs dx, dy
            if names[k] == "Pt_to_VolAve_2D" or names[k] == "VoleAve_to_pt" or \
               names[k] == "VoleAve_to_dxdy" or names[k] == "VoleAve_to_dxdx" or \
               names[k] == "VoleAve_to_dydy" or names[k] == "VoleAve_to_dx" or \
               names[k] == "VoleAve_to_dy": 
                # Assuming these names correspond to scenarios where exact_sols[k] is exact_volume_average_f_grid
                # or if it's a point function that needs to be evaluated on volume average inputs (which seems unlikely here based on name)
                # For "Pt_to_VolAve_2D", exact_sols[k] IS exact_volume_average_f_grid
                if callable(exact_sols[k]) and exact_sols[k].__name__ == 'exact_volume_average_f_grid':
                    exact_sol = exact_sols[k](X_predict, Y_predict, g2d.dx, g2d.dy).flatten()
                else:
                    # This case would be if an exact solution like 'f' was being compared to a GP prediction of volume average.
                    # This interpretation might be mixed. The primary use here is for exact_volume_average_f_grid itself.
                    # If the GP *output* is volume average, the exact solution is volume average.
                    # If the GP *input* is volume average but *output* is point, exact solution is point.
                    # The current logic seems to be that exact_sols[k] IS the function to get the true values for the comparison.
                    # So if names[k] indicates volume average, exact_sols[k] should be exact_volume_average_f_grid.
                    exact_sol = exact_sols[k](X_predict, Y_predict).flatten() # Fallback, review if this logic is correct for all named cases
            else:
                 exact_sol = exact_sols[k](X_predict, Y_predict).flatten()


            #L1 ERROR
            #err = np.linalg.norm(exact_sol-  sol, ord=1)
            err = error.L1_error(exact_sol, sol, g2d.dx*g2d.dy)
            if high_precision:
                error_l1[i,j] = err
            else:
                error_l1[i][j] = err
                            
            #L2 ERROR
            #err = np.linalg.norm(exact_sol-  sol, ord=2)
            err = error.L2_error(exact_sol, sol, g2d.dx*g2d.dy)

            if high_precision:
                error_l2[i,j] = err
            else:
                error_l2[i][j] = err


            #LInfinity ERROR
            err = error.L_infinity_error(exact_sol, sol)
            if high_precision:
                error_linf[i,j] = err
            else:
                error_linf[i][j] = err

            # Plotting for debugging...
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            # # Plot the training data on the leftmost pane
            # contour_train = ax1.contourf(X, Y, g2d.grid, cmap="viridis")
            # ax1.set_title(f"Training Data for {names[k]}")
            # fig.colorbar(contour_train, ax=ax1, label="Training Function")

            # # Plot the GP solution on the middle pane
            # contour_gp = ax2.contourf(X_predict, Y_predict, sol.reshape(X_predict.shape), cmap="viridis")
            # ax2.set_title(f"GP Solution for {names[k]} R_GP = {r_gp}")
            # fig.colorbar(contour_gp, ax=ax2, label="GP sol")

            # # Plot the exact solution on the rightmost pane
            # contour_exact = ax3.contourf(X_predict, Y_predict, exact_sol.reshape(X_predict.shape), cmap="viridis")
            # ax3.set_title(f"Exact Solution for {names[k]} R_GP = {r_gp}")
            # fig.colorbar(contour_exact, ax=ax3, label="Exact Sol")

            # plt.tight_layout()
            # plt.show()



    # #Save Plot.
    # colors = ['thistle', 'darkslategrey', 'mediumseagreen', 'orange', 'purple', 'black']
    # for i, r_gp in enumerate(r_gps):
        
    #     ax1.loglog(Nxs, error_l1[i], label="r_gp = " + str(r_gp), c=colors[i], linestyle='-', marker='s')

    #     # Compute the theoretical convergence line using mpmath
    #     conv_line = []
    #     for N in Nxs:
    #         conv_line.append(float(mm.mpf(error_l1[i][0]) * mm.power(Nxs[0], 2*r_gp+1) / mm.power(N, 2*r_gp+1)))
    #     conv_line = np.array(conv_line)

    #     ax1.loglog(Nxs, conv_line, '--', c=colors[i], alpha=0.5,label="Convergence Line O({})".format(2*r_gp+1))

    # ax1.set_title("Convergence Plot {}".format(names[k]))
    # ax1.set_ylabel("L1 Error")
    # ax1.set_xlabel("Grid Points (Nx)")
    # ax1.legend(loc='center left', bbox_to_anchor=(1, .6))
    # plt.tight_layout()
    # plt.savefig("convergence_tables/{}.png".format(names[k]), dpi=400)

    #Print Table.
    print("========={}==============".format(names[k]))

 

    for i, r_gp in enumerate(r_gps):
        print("----r_gp = {}----".format(r_gp))

        for j in range(len(Nxs)):

            if j == 0: #first pass, we don't have EOC.
                if high_precision:
                    print(f"Nx: {Nxs[j]}  L1:  {float(error_l1[i,j]):.4e}   EOC: ----    L2: {float(error_l2[i,j]):.4e}   EOC: ---")
                else:
                    print(f"Nx: {Nxs[j]}  L1:  {error_l1[i][j]:.4e}   EOC: ----    L2: {error_l2[i][j]:.4e}   EOC: ---")
            else:
                if high_precision:
                    EOC_L1 = mm.log(error_l1[i,j-1]/ error_l1[i,j])/mm.log(2)
                    EOC_L2 = mm.log(error_l2[i,j-1]/ error_l2[i,j])/mm.log(2)
                    EOC_Linf = mm.log(error_linf[i,j-1]/ error_linf[i,j])/mm.log(2)
                    EOC_m_L1[i,j] = EOC_L1
                    EOC_m_L2[i,j] = EOC_L2
                    EOC_m_Linfty[i,j] = EOC_Linf

                    #print(f"Nx: {Nxs[j]}  L1:  {float(error_l1[i,j]):.4e}   EOC: {float(EOC_L1):.2f}    L2: {float(error_l2[i,j]):.4e}   EOC: {float(EOC_L2):.2f}")

                else:
                    EOC_L1 = np.log(error_l1[i][j-1]/ error_l1[i][j])/np.log(2)
                    EOC_L2 = np.log(error_l2[i][j-1]/ error_l2[i][j])/np.log(2)
                    EOC_Linf = np.log(error_linf[i][j-1]/ error_linf[i][j])/np.log(2)

                    #print(f"Nx: {Nxs[j]}  L1:  {error_l1[i][j]:.4e}   EOC: {EOC_L1:.2f}    L2: {error_l2[i][j]:.4e}   EOC: {EOC_L2:.2f}")

                    EOC_m_L1[i][j] = EOC_L1
                    EOC_m_L2[i][j] = EOC_L2
                    EOC_m_Linfty[i][j] = EOC_Linf

    print("===latex===")
    latex_str = rf'''\begin{{table}}[ht!]
        \footnotesize
        \centering
        \caption{{names[k]}}
        \label{{"table:" + names[k]}}
        \begin{{tabular}}{{@{{}}cccc|ccc|cccc@{{}}}}
            \toprule
            \multirow{{2}}{{*}}{{Grid Res.}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 1$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 2$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 3$}} \\ 
            \cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}} \cmidrule(lr){{8-10}} 
            & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error \\
            \midrule
            \( 25 \)   & {float(error_l1[0,0]):.4e} & {float(error_l2[0,0]):.4e} & {float(error_linf[0,0]):.4e} & {float(error_l1[1,0]):.4e} & {float(error_l2[1,0]):.4e} & {float(error_linf[1,0]):.4e} & {float(error_l1[2,0]):.4e} & {float(error_l2[2,0]):.4e} & {float(error_linf[2,0]):.4e} \\
            \( 50 \)   & {float(error_l1[0,1]):.4e} & {float(error_l2[0,1]):.4e} & {float(error_linf[0,1]):.4e} & {float(error_l1[1,1]):.4e} & {float(error_l2[1,1]):.4e} & {float(error_linf[1,1]):.4e} & {float(error_l1[2,1]):.4e} & {float(error_l2[2,1]):.4e} & {float(error_linf[2,1]):.4e} \\
            \( 100 \)  & {float(error_l1[0,2]):.4e} & {float(error_l2[0,2]):.4e} & {float(error_linf[0,2]):.4e} & {float(error_l1[1,2]):.4e} & {float(error_l2[1,2]):.4e} & {float(error_linf[1,2]):.4e} & {float(error_l1[2,2]):.4e} & {float(error_l2[2,2]):.4e} & {float(error_linf[2,2]):.4e} \\
            \( 200 \)  & {float(error_l1[0,3]):.4e} & {float(error_l2[0,3]):.4e} & {float(error_linf[0,3]):.4e} & {float(error_l1[1,3]):.4e} & {float(error_l2[1,3]):.4e} & {float(error_linf[1,3]):.4e} & {float(error_l1[2,3]):.4e} & {float(error_l2[2,3]):.4e} & {float(error_linf[2,3]):.4e} \\
            \( 400 \)  & {float(error_l1[0,4]):.4e} & {float(error_l2[0,4]):.4e} & {float(error_linf[0,4]):.4e} & {float(error_l1[1,4]):.4e} & {float(error_l2[1,4]):.4e} & {float(error_linf[1,4]):.4e} & {float(error_l1[2,4]):.4e} & {float(error_l2[2,4]):.4e} & {float(error_linf[2,4]):.4e} \\

            \midrule
            & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC \\
            \midrule
            \( 25 \)   & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} \\
            \( 50 \)   & {float(EOC_m_L1[0,1]):.2f} & {float(EOC_m_L2[0,1]):.2f} & {float(EOC_m_Linfty[0,1]):.2f} & {float(EOC_m_L1[1,1]):.2f} & {float(EOC_m_L2[1,1]):.2f} & {float(EOC_m_Linfty[1,1]):.2f} & {float(EOC_m_L1[2,1]):.2f} & {float(EOC_m_L2[2,1]):.2f} & {float(EOC_m_Linfty[2,1]):.2f} \\
            \( 100 \)  & {float(EOC_m_L1[0,2]):.2f} & {float(EOC_m_L2[0,2]):.2f} & {float(EOC_m_Linfty[0,2]):.2f} & {float(EOC_m_L1[1,2]):.2f} & {float(EOC_m_L2[1,2]):.2f} & {float(EOC_m_Linfty[1,2]):.2f} & {float(EOC_m_L1[2,2]):.2f} & {float(EOC_m_L2[2,2]):.2f} & {float(EOC_m_Linfty[2,2]):.2f} \\
            \( 200 \)  & {float(EOC_m_L1[0,3]):.2f} & {float(EOC_m_L2[0,3]):.2f} & {float(EOC_m_Linfty[0,3]):.2f} & {float(EOC_m_L1[1,3]):.2f} & {float(EOC_m_L2[1,3]):.2f} & {float(EOC_m_Linfty[1,3]):.2f} & {float(EOC_m_L1[2,3]):.2f} & {float(EOC_m_L2[2,3]):.2f} & {float(EOC_m_Linfty[2,3]):.2f} \\
            \( 400 \)  & {float(EOC_m_L1[0,4]):.2f} & {float(EOC_m_L2[0,4]):.2f} & {float(EOC_m_Linfty[0,4]):.2f} & {float(EOC_m_L1[1,4]):.2f} & {float(EOC_m_L2[1,4]):.2f} & {float(EOC_m_Linfty[1,4]):.2f} & {float(EOC_m_L1[2,4]):.2f} & {float(EOC_m_L2[2,4]):.2f} & {float(EOC_m_Linfty[2,4]):.2f} \\

            \bottomrule
        \end{{tabular}}
    \end{{table}}'''
    print(latex_str)

    
    #print out time script took to run.
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))



