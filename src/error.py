import numpy as np
import mpmath as mm

def L2_error(u_sol, u_pred, dx):

    error_l2 = 0
    for i in range(len(u_sol)):
        error_l2 += (u_sol[i] - u_pred[i])**2

    return np.sqrt(dx*error_l2)

def L2_error_2D(u_sol, u_pred, dx,dy):

    error_l2 = 0
    for i in range(len(u_sol)):
        for j in range(len(u_sol[i])):
            error_l2 += (u_sol[i,j] - u_pred[i,j])**2

    return np.sqrt(dx*dy*error_l2)


def L1_error(u_sol, u_pred, dx):

    error_l1 = 0
    for i in range(len(u_sol)):
        error_l1 += np.sqrt((u_sol[i] - u_pred[i])**2) #mpmath doesn't have an abs....

    return dx*error_l1

def L1_error_2D(u_sol, u_pred, dx,dy):

    error_l1 = 0
    for i in range(len(u_sol)):
        for j in range(len(u_sol[i])):
            error_l1 += np.sqrt((u_sol[i,j] - u_pred[i,j])**2) #mpmath doesn't have an abs....

    return dx*dy*error_l1


def L_infinity_error(u_sol, u_pred):
    error_inf = 0
    for i in range(len(u_sol)):
        error = np.sqrt((u_sol[i] - u_pred[i])**2)  # Using mpmath sqrt to mimic absolute function.
        if error > error_inf:
            error_inf = error

    return error_inf

def L_infinity_error_2D(u_sol, u_pred):
    error_inf = 0
    for i in range(len(u_sol)):
        for j in range(len(u_sol[i])):
            error = np.sqrt((u_sol[i,j] - u_pred[i,j])**2)  # Using mpmath sqrt to mimic absolute function.
            if error > error_inf:
                error_inf = error

    return error_inf
