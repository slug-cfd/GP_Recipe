#Showcase 2D.
#Take laplacian in 2D.

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
from mpl_toolkits.mplot3d import Axes3D

# mm.mp.dps = 1000

#INPUTS
Nx = 128
Ny = 128
r_gp = 2
xlim = (0,2*np.pi)
ylim = (0,2*np.pi)
high_precision=True

plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

#function that assigns the grid data.
exact_f = lambda X,Y : np.sin(X) * np.sin(Y)

#EXACT SOLUTIONS
f = lambda X,Y : np.sin(X) * np.sin(Y)

#laplacian
L_lap_np = lambda X,Y: -2 * np.sin(X) * np.sin(Y)


# Laplacian 
L_lap = sym.diff(kern.SE_2d, kern.x_sym1, kern.x_sym1) + sym.diff(kern.SE_2d, kern.y_sym1, kern.y_sym1)

g2d = Grid2D(xlim, ylim, Nx, Ny, r_gp)
g2d.fill_grid(f)

gp_recipe2d = GP_recipe2D(g2d, r_gp,  ell=np.pi/10, stencil_method="blocky_diamond", high_precision=high_precision, mean_method="zero", precision=113)

X, Y = np.meshgrid(g2d.x, g2d.y)
exact_sol = f(X,Y)

x_predict = g2d.x_int + g2d.dx * 0.5
x_predict = x_predict[:-1]
y_predict = g2d.y_int+ g2d.dx * 0.5
y_predict = y_predict[:-1]



X_predict, Y_predict = np.meshgrid(x_predict, y_predict)

points_predict = [[x, y] for x, y in zip(X_predict.flatten(), Y_predict.flatten())]
sol_SE = gp_recipe2d.convert_custom(points_predict, kern.SE_2d, L_lap, stationary=True)

sol_SE = sol_SE.reshape(X_predict.shape)

exact_sol = L_lap_np(X_predict,Y_predict)



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X_predict, Y_predict, f(X_predict,Y_predict), color='green', alpha=0.3)
# ax.plot_surface(X_predict, Y_predict, exact_sol, color='blue', alpha=0.3)

# ax.plot_surface(X_predict, Y_predict, sol_SE, color='red', alpha=0.8)

# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# ax.set_title("SE Kernel")
# plt.show()



# Create a figure
fig = plt.figure(figsize=(18, 6))

# Add first subplot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_predict, Y_predict, f(X_predict,Y_predict), cmap='viridis')
ax1.set_title('Initial Condition')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_zlabel('$f(x,y)$')

# Add second subplot
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X_predict, Y_predict, exact_sol, cmap='plasma')
ax2.set_title('GP Prediction')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.set_zlabel('$f(x,y)$')

# Add third subplot
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X_predict, Y_predict, sol_SE, cmap='plasma')
ax3.set_title('Exact Solution')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$y$')
ax3.set_zlabel('$f(x,y)$')


# Display the plot
# fig.tight_layout()
plt.savefig("final_plots/2D_showcase.png")
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax.plot_surface(X_predict, Y_predict, exact_sol, color='blue', alpha=0.3)
# ax.plot_surface(X_predict, Y_predict, exact_sol, color='red', alpha=0.8)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# ax.set_title("Exact Solution")
# plt.show(block=False)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax.plot_surface(X_predict, Y_predict, exact_sol, color='blue', alpha=0.3)
# ax.plot_surface(X_predict, Y_predict, np.abs(exact_sol-sol_SE), color='red', alpha=0.8)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# ax.set_title("Error")
# plt.show()

