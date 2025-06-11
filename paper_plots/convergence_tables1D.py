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
sys.path.append('../../')
import sympy as sym
from driver import GP_driver_data
from new_driver import GP_recipe1D, GP_recipe2D
import inspect
import numpy as np
import matplotlib.pyplot as plt
from grids import Grid1D, Grid2D
import kernels as kern
import time
import mpmath as mm
start_time = time.time()
import error
import sympy as sym
PI = 3.14159
TEST = False
PLOT = False

NEW_FUNCTIONS = True #if true, use complicated new functions
precision_num = 113


def convergence_tables1D():

    with open('convergence_tables1D.txt', 'w') as file:

        if TEST:
            r_gps = [1, 2]
            Nxs = np.array([16,32, 64, 128])
        else:
            r_gps = [1, 2, 3]
            Nxs = np.array([16,32, 64, 128, 256])

        #function that assigns the grid data.
        #exact_f = lambda x : np.sin(2*PI*x)

        def shock(x):
            y = np.zeros_like(x)

            for i in range(len(x)):

                if x[i] < 1:
                    y[i] = 0
                else:
                    y[i] = 1
            return y

        high_precision = True

        #EXACT SOLUTIONS

        if NEW_FUNCTIONS:
            f = lambda x: np.sin(4*np.pi*x) * np.cos(2 * np.pi * x) * np.exp( -x)

            x_sym = sym.symbols('x')
            pi_sym = sym.pi  # Use symbolic pi for precision

            # Define the symbolic function
            f_sym = sym.sin(4*np.pi*x_sym) * sym.cos(2 * pi_sym * x_sym) * sym.exp(
                - x_sym)

            # Compute derivatives symbolically
            ddx_sym = sym.diff(f_sym, x_sym, 1)
            ddxx_sym = sym.diff(f_sym, x_sym, 2)
            ddxxx_sym = sym.diff(f_sym, x_sym, 3)

            # Compute integral symbolically
            integral_sym = sym.integrate(f_sym, x_sym)

            # Convert symbolic expressions to numpy lambda functions
            # Using 'numpy' module allows the resulting functions to operate on numpy arrays
            ddx = sym.lambdify(x_sym,
                               ddx_sym,
                               modules=['numpy', {
                                   'pi': np.pi
                               }])
            ddxx = sym.lambdify(x_sym,
                                ddxx_sym,
                                modules=['numpy', {
                                    'pi': np.pi
                                }])
            ddxxx = sym.lambdify(x_sym,
                                 ddxxx_sym,
                                 modules=['numpy', {
                                     'pi': np.pi
                                 }])
            # Define the integral function using lambdify
            integral = sym.lambdify(x_sym,
                                    integral_sym,
                                    modules=['numpy', {
                                        'pi': np.pi
                                    }])

            def volume_average(x):
                dx = x[1] - x[0]
                y = np.zeros_like(x)
                for i in range(len(x)):
                    y[i] = (integral(x[i] + dx / 2) -
                            integral(x[i] - dx / 2)) / dx
                return y

        else:
            f = lambda x: np.sin(2 * PI * x)
            ddx = lambda x: np.cos(2 * PI * x) * 2 * PI
            ddxx = lambda x: -np.sin(2 * PI * x) * (2 * PI)**2
            ddxxx = lambda x: -np.cos(2 * PI * x) * (2 * PI)**3
            integral = lambda x: -np.cos(2 * PI * x) / (2 * PI)

            def volume_average(x):
                f = lambda x: -np.cos(2 * PI * x) / (2 * PI)
                dx = x[1] - x[0]
                y = np.zeros_like(x)

                for i in range(len(x)):
                    y[i] = (f(x[i] + dx / 2) - f(x[i] - dx / 2)) / dx
                return y

        exact_sols = [
            f, ddx, ddxx, ddxxx, volume_average, integral, f, f, shock, shock,
            ddx, f, f
        ]
        names = [
            "Pt_to_Pt", "Pt_to_Der", "Pt_to_2Der", "Pt_to_3der",
            "Pt_to_VolAve", "Pt_to_Int", "Pt_to_Pt_NN", "Pt_to_Pt_AS",
            "Pt_to_Pt_SE-SHOCK", "Pt_to_Pt_AS-SHOCK", "Pt_to_Der_AS",
            "Pt_to_Pt_DASv9", "VolAve_to_Pt"
        ]
        dims = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        kint = sym.integrate(kern.SE, kern.x_sym)
        left_integral = kint.subs(kern.x_sym, kern.x_sym - kern.dx_sym / 2)
        right_integral = kint.subs(kern.x_sym, kern.x_sym + kern.dx_sym / 2)
        T = (right_integral - left_integral) / kern.dx_sym

        # print("T:")
        # print(T)

        # print("making tas")
        # kint = sym.integrate(kern.AS, kern.x_sym)
        # print(kint)
        # left_integral = kint.subs(kern.x_sym, kern.x_sym-kern.dx_sym/2)
        # print(left_integral)
        # right_integral = kint.subs(kern.x_sym, kern.x_sym+kern.dx_sym/2)
        # print(right_integral)
        # T_AS = (right_integral - left_integral)/kern.dx_sym
        # print(T_AS)

        f1 = sym.integrate(kern.SE, kern.x_sym)
        x_integral = (
            f1.subs(kern.x_sym, kern.x_sym + kern.dx_sym / 2) -
            f1.subs(kern.x_sym, kern.x_sym - kern.dx_sym / 2)) / kern.dx_sym
        f2 = sym.integrate(x_integral, kern.y_sym)
        f2.simplify()
        C = (f2.subs(kern.y_sym, kern.y_sym + kern.dx_sym / 2) -
             f2.subs(kern.y_sym, kern.y_sym - kern.dx_sym / 2)) / kern.dx_sym
        C = C.simplify()
        # print("C: ", C)

        # print("making cas")

        # f1 = sym.integrate(kern.AS, kern.x_sym)
        # x_integral = (f1.subs(kern.x_sym, kern.x_sym + kern.dx_sym/2) - f1.subs(kern.x_sym, kern.x_sym - kern.dx_sym/2))/kern.dx_sym
        # f2 = sym.integrate(x_integral, kern.y_sym)
        # f2.simplify()
        # C_AS = (f2.subs(kern.y_sym, kern.y_sym + kern.dx_sym/2) - f2.subs(kern.y_sym, kern.y_sym - kern.dx_sym/2))/kern.dx_sym
        # C_AS = C_AS.simplify()

        # print("done")

        k_kernels = [
            kern.SE, kern.SE, kern.SE, kern.SE, kern.SE, kern.SE, kern.AS2,
            kern.AS, kern.AS, kern.SE, kern.AS, kern.DAS_V9, C
        ]
        k_star_kernels = [
            kern.SE,
            sym.diff(kern.SE, kern.y_sym, 1),
            sym.diff(kern.SE, kern.y_sym, 2),
            sym.diff(kern.SE, kern.y_sym, 3), T, C, kern.AS2, kern.AS, kern.AS,
            kern.SE,
            sym.diff(kern.AS, kern.y_sym, 1), kern.DAS_V9, T
        ]

        assert (len(names) == len(exact_sols))
        assert (len(names) == len(k_kernels))
        assert (len(names) == len(k_star_kernels))
        assert (len(names) == len(dims))

        #for k in range(len(names)-2): #loop over each type of conversion.

        conversions_for_paper = [0, 1, 2, 3, 4, 5, 6, 11, 12]
        # conversions_for_paper = [0]

        for k in conversions_for_paper:
            print(names[k])

            n_rgps = 3
            n_Nxs = 6

            error_l1 = np.zeros((n_rgps, n_Nxs))
            error_l2 = np.zeros((n_rgps, n_Nxs))
            error_linf = np.zeros((n_rgps, n_Nxs))
            EOC_m_L1 = np.zeros((n_rgps, n_Nxs))
            EOC_m_L2 = np.zeros((n_rgps, n_Nxs))
            EOC_m_Linfty = np.zeros((n_rgps, n_Nxs))

            if high_precision:
                error_l1 = mm.matrix(error_l1.tolist())
                error_l2 = mm.matrix(error_l2.tolist())
                error_linf = mm.matrix(error_linf.tolist())
                EOC_m_L1 = mm.matrix(EOC_m_L1.tolist())
                EOC_m_L2 = mm.matrix(EOC_m_L2.tolist())
                EOC_m_Linfty = mm.matrix(EOC_m_Linfty.tolist())

            fig, (ax1) = plt.subplots(1, 1, sharey=True)

            for i, r_gp in enumerate(r_gps):
                for j, Nx in enumerate(Nxs):

                    xlim = (0, 1)

                    g = Grid1D(xlim, Nx, r_gp)
                    if names[k] == "VolAve_to_Pt":
                        g.fill_grid(volume_average)
                    else:
                        g.fill_grid(f)
                    dx = g.x[1] - g.x[0]

                    #keep ell/dx constant.
                    #ell = 1.2 for Nx = 25 works well.
                    # ell = (1.2/((xlim[1]-xlim[0])/25))*dx
                    # print("Nx: ", Nx, " ell: ", ell)
                    #ell = 0.8
                    ell = 0.05

                    gprecipe = GP_recipe1D(
                        g,
                        r_gp,
                        ell=ell,
                        stencil_method="center",
                        high_precision=high_precision,
                        precision=precision_num)

                    x_predict = g.x_int + dx / 2
                    x_predict = x_predict[1:]

                    y_predict = gprecipe.convert_custom(x_predict,
                                                        k_kernels[k],
                                                        k_star_kernels[k],
                                                        stationary=False)

                    if high_precision:
                        y_predict = mm.matrix(y_predict.tolist())

                    y_exact = exact_sols[k](x_predict)
                    #print(type(y_exact))
                    #mp_matrix = mm.matrix(y_exact.tolist())

                    #L1 ERROR
                    #err = np.linalg.norm(y_exact-  y_predict, ord=1)
                    err = error.L1_error(y_exact, y_predict, dx)

                    if high_precision:
                        error_l1[i, j] = err
                    else:
                        error_l1[i][j] = err

                    #L2 ERROR
                    #err = np.linalg.norm(y_exact-  y_predict, ord=2)
                    err = error.L2_error(y_exact, y_predict, dx)

                    if high_precision:
                        error_l2[i, j] = err
                    else:
                        error_l2[i][j] = err

                    #LInfinity ERROR
                    err = error.L_infinity_error(y_exact, y_predict)
                    if high_precision:
                        error_linf[i, j] = err
                    else:
                        error_linf[i][j] = err

                    #plotting for debugging...
                    # plt.figure()
                    # plt.plot(g.x, g.grid, label="Training Function")
                    # plt.plot(x_predict, y_predict, '.',label="GP sol")
                    # plt.plot(x_predict, y_exact, '.', label="Exact Sol")
                    # plt.title(f"{names[k]}  R_GP = {r_gp}")
                    # plt.legend()
                    # plt.show()

            #Save Plot.
            if PLOT:
                colors = [
                    'thistle', 'darkslategrey', 'mediumseagreen', 'orange',
                    'purple', 'black'
                ]
                for i, r_gp in enumerate(r_gps):

                    if high_precision:
                        ax1.loglog(Nxs,
                                   error_l1[i, :],
                                   label="r_gp = " + str(r_gp),
                                   c=colors[i],
                                   linestyle='-',
                                   marker='s')
                    else:
                        ax1.loglog(Nxs,
                                   error_l1[i],
                                   label="r_gp = " + str(r_gp),
                                   c=colors[i],
                                   linestyle='-',
                                   marker='s')

                    # Compute the theoretical convergence line using mpmath
                    conv_line = []
                    for N in Nxs:
                        if high_precision:
                            conv_line.append(
                                float(
                                    mm.mpf(error_l1[i, 0]) *
                                    mm.power(Nxs[0], 2 * r_gp + 1) /
                                    mm.power(N, 2 * r_gp + 1)))
                        else:
                            conv_line.append(
                                float(
                                    mm.mpf(error_l1[i][0]) *
                                    mm.power(Nxs[0], 2 * r_gp + 1) /
                                    mm.power(N, 2 * r_gp + 1)))

                    conv_line = np.array(conv_line)

                    ax1.loglog(Nxs,
                               conv_line,
                               '--',
                               c=colors[i],
                               alpha=0.5,
                               label="Convergence Line O({})".format(2 * r_gp +
                                                                     1))

                ax1.set_title("Convergence Plot {}".format(names[k]))
                ax1.set_ylabel("L1 Error")
                ax1.set_xlabel("Grid Points (Nx)")
                ax1.legend(loc='center left', bbox_to_anchor=(1, .6))
                plt.tight_layout()
                plt.savefig("convergence_tables/{}.png".format(names[k]),
                            dpi=400)

            #Print Table.
            print("========={}==============".format(names[k]), file=file)

            for i, r_gp in enumerate(r_gps):
                print("----r_gp = {}----".format(r_gp), file=file)

                for j in range(len(Nxs)):

                    if j == 0:  #first pass, we don't have EOC.
                        if high_precision:
                            print(
                                f"Nx: {Nxs[j]}  L1:  {float(error_l1[i,j]):.4e}   EOC: ----    L2: {float(error_l2[i,j]):.4e}   EOC: ---      L_INF: {float(error_linf[i,j]):.4e}   EOC: ---   ",
                                file=file)

                        else:
                            print(
                                f"Nx: {Nxs[j]}  L1:  {error_l1[i][j]:.4e}   EOC: ----    L2: {error_l2[i][j]:.4e}   EOC: ---",
                                file=file)

                    else:
                        if high_precision:
                            EOC_L1 = mm.log(error_l1[i, j - 1] /
                                            error_l1[i, j]) / mm.log(2)
                            EOC_L2 = mm.log(error_l2[i, j - 1] /
                                            error_l2[i, j]) / mm.log(2)
                            EOC_Linf = mm.log(error_linf[i, j - 1] /
                                              error_linf[i, j]) / mm.log(2)

                            print("EOC_L1: ", EOC_L1)
                            print("EOC_L2: ", EOC_L2)

                            print("EOC_Linf: ", EOC_Linf)

                            # print(float(error_l1[i,j]))
                            # print(EOC_L1)
                            #try:
                            #error_l1[i,j]
                            print(
                                f"Nx: {Nxs[j]}  L1:  {float(error_l1[i,j]):.4e}   EOC: {float(EOC_L1):.3f}    L2: {float(error_l2[i,j]):.4e}   EOC: {float(EOC_L2):.3f}      L_INF: {float(error_linf[i,j]):.4e}   EOC: {float(EOC_Linf):.3f}   ",
                                file=file)
                            # except:
                            #     print(f"Nx: {Nxs[j]}  L1:  {error_l1[i,j]}   EOC: {EOC_L1)}    L2: {error_l2[i,j]):.4e}   EOC: {float(EOC_L2):.3f}      L_INF: {float(error_linf[i,j]):.4e}   EOC: {float(EOC_Linf):.3f}   ", file=file)

                            EOC_m_L1[i, j] = EOC_L1
                            EOC_m_L2[i, j] = EOC_L2
                            EOC_m_Linfty[i, j] = EOC_Linf

                        else:
                            EOC_L1 = np.log(error_l1[i][j - 1] /
                                            error_l1[i][j]) / np.log(2)
                            EOC_L2 = np.log(error_l2[i][j - 1] /
                                            error_l2[i][j]) / np.log(2)
                            EOC_Linf = np.log(error_linf[i][j - 1] /
                                              error_linf[i][j]) / np.log(2)
                            EOC_m_L1[i][j] = EOC_L1
                            EOC_m_L2[i][j] = EOC_L2
                            EOC_m_Linfty[i][j] = EOC_Linf

                            print(
                                f"Nx: {Nxs[j]}  L1:  {error_l1[i][j]:.4e}   EOC: {EOC_L1:.3f}    L2: {error_l2[i][j]:.4e}   EOC: {EOC_L2:.3f}    L_INF: {error_linf[i][j]:.4e}   EOC: {EOC_Linf:.3f}",
                                file=file)

                            # print("====")
                            # print("EOC_L1 setting: ", EOC_m_L1[i][j], " for i=", i, " j=",j)
                            # print("EOC_L1 setting: ", EOC_L1, " for i=", i, " j=",j)
                            # print("error_l1[i][j-1]: ", error_l1[i][j-1])
                            # print("error_l1[i][j]: ", error_l1[i][j])
                            # print("error_l1[i][j-1]/ error_l1[i][j]: ", error_l1[i][j-1]/ error_l1[i][j])
                            EOC_L1 = np.log(error_l1[i][j - 1] /
                                            error_l1[i][j]) / np.log(2)
                            #print("EOC_L1:", EOC_L1)

            print("===latex===", file=file)
            latex_str = rf'''\begin{{table}}[ht!]
                    \footnotesize
                    \centering
                    \caption{{\text{names[k]}}}
                    \label{{table:Pt_to_Pt_1D}}
                    \begin{{tabular}}{{@{{}}cccc|ccc|cccc@{{}}}}
                        \toprule
                        \multirow{{2}}{{*}}{{Grid Res.}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 1$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 2$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 3$}} \\ 
                        \cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}} \cmidrule(lr){{8-10}} 
                        & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error \\
                        \midrule
                        \( 16 \)   & {float(error_l1[0,0]):.4e} & {float(error_l2[0,0]):.4e} & {float(error_linf[0,0]):.4e} & {float(error_l1[1,0]):.4e} & {float(error_l2[1,0]):.4e} & {float(error_linf[1,0]):.4e} & {float(error_l1[2,0]):.4e} & {float(error_l2[2,0]):.4e} & {float(error_linf[2,0]):.4e} \\
                        \( 32 \)   & {float(error_l1[0,1]):.4e} & {float(error_l2[0,1]):.4e} & {float(error_linf[0,1]):.4e} & {float(error_l1[1,1]):.4e} & {float(error_l2[1,1]):.4e} & {float(error_linf[1,1]):.4e} & {float(error_l1[2,1]):.4e} & {float(error_l2[2,1]):.4e} & {float(error_linf[2,1]):.4e} \\
                        \( 64 \)  & {float(error_l1[0,2]):.4e} & {float(error_l2[0,2]):.4e} & {float(error_linf[0,2]):.4e} & {float(error_l1[1,2]):.4e} & {float(error_l2[1,2]):.4e} & {float(error_linf[1,2]):.4e} & {float(error_l1[2,2]):.4e} & {float(error_l2[2,2]):.4e} & {float(error_linf[2,2]):.4e} \\
                        \( 128 \)  & {float(error_l1[0,3]):.4e} & {float(error_l2[0,3]):.4e} & {float(error_linf[0,3]):.4e} & {float(error_l1[1,3]):.4e} & {float(error_l2[1,3]):.4e} & {float(error_linf[1,3]):.4e} & {float(error_l1[2,3]):.4e} & {float(error_l2[2,3]):.4e} & {float(error_linf[2,3]):.4e} \\
                        \( 256 \)  & {float(error_l1[0,4]):.4e} & {float(error_l2[0,4]):.4e} & {float(error_linf[0,4]):.4e} & {float(error_l1[1,4]):.4e} & {float(error_l2[1,4]):.4e} & {float(error_linf[1,4]):.4e} & {float(error_l1[2,4]):.4e} & {float(error_l2[2,4]):.4e} & {float(error_linf[2,4]):.4e} \\
                        \midrule
                        & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC \\
                        \midrule
                        \( 16 \)   & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} \\
                        \( 32 \)   & {float(EOC_m_L1[0,1]):.3f} & {float(EOC_m_L2[0,1]):.3f} & {float(EOC_m_Linfty[0,1]):.3f} & {float(EOC_m_L1[1,1]):.3f} & {float(EOC_m_L2[1,1]):.3f} & {float(EOC_m_Linfty[1,1]):.3f} & {float(EOC_m_L1[2,1]):.3f} & {float(EOC_m_L2[2,1]):.3f} & {float(EOC_m_Linfty[2,1]):.3f} \\
                        \( 64 \)  & {float(EOC_m_L1[0,2]):.3f} & {float(EOC_m_L2[0,2]):.3f} & {float(EOC_m_Linfty[0,2]):.3f} & {float(EOC_m_L1[1,2]):.3f} & {float(EOC_m_L2[1,2]):.3f} & {float(EOC_m_Linfty[1,2]):.3f} & {float(EOC_m_L1[2,2]):.3f} & {float(EOC_m_L2[2,2]):.3f} & {float(EOC_m_Linfty[2,2]):.3f} \\
                        \( 128 \)  & {float(EOC_m_L1[0,3]):.3f} & {float(EOC_m_L2[0,3]):.3f} & {float(EOC_m_Linfty[0,3]):.3f} & {float(EOC_m_L1[1,3]):.3f} & {float(EOC_m_L2[1,3]):.3f} & {float(EOC_m_Linfty[1,3]):.3f} & {float(EOC_m_L1[2,3]):.3f} & {float(EOC_m_L2[2,3]):.3f} & {float(EOC_m_Linfty[2,3]):.3f} \\
                        \( 256 \)  & {float(EOC_m_L1[0,4]):.3f} & {float(EOC_m_L2[0,4]):.3f} & {float(EOC_m_Linfty[0,4]):.3f} & {float(EOC_m_L1[1,4]):.3f} & {float(EOC_m_L2[1,4]):.3f} & {float(EOC_m_Linfty[1,4]):.3f} & {float(EOC_m_L1[2,4]):.3f} & {float(EOC_m_L2[2,4]):.3f} & {float(EOC_m_Linfty[2,4]):.3f} \\

                        \bottomrule
                    \end{{tabular}}
                \end{{table}}'''

            # latex_str = rf'''\begin{{table}}[ht!]
            #     \footnotesize
            #     \centering
            #     \caption{{names[k]}}
            #     \label{{table:Pt_to_Pt_1D}}
            #     \begin{{tabular}}{{@{{}}cccc|ccc|cccc@{{}}}}
            #         \toprule
            #         \multirow{{2}}{{*}}{{Grid Res.}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 1$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 2$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 3$}} \\
            #         \cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}} \cmidrule(lr){{8-10}}
            #         & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error \\
            #         \midrule
            #         \( 25 \)   & {float(error_l1[0,0]):.4e} & {float(error_l2[0,0]):.4e} & {float(error_linf[0,0]):.4e} & {float(error_l1[1,0]):.4e} & {float(error_l2[1,0]):.4e} & {float(error_linf[1,0]):.4e} & {float(error_l1[2,0]):.4e} & {float(error_l2[2,0]):.4e} & {float(error_linf[2,0]):.4e} \\
            #         \( 50 \)   & {float(error_l1[0,1]):.4e} & {float(error_l2[0,1]):.4e} & {float(error_linf[0,1]):.4e} & {float(error_l1[1,1]):.4e} & {float(error_l2[1,1]):.4e} & {float(error_linf[1,1]):.4e} & {float(error_l1[2,1]):.4e} & {float(error_l2[2,1]):.4e} & {float(error_linf[2,1]):.4e} \\
            #         \( 100 \)  & {float(error_l1[0,2]):.4e} & {float(error_l2[0,2]):.4e} & {float(error_linf[0,2]):.4e} & {float(error_l1[1,2]):.4e} & {float(error_l2[1,2]):.4e} & {float(error_linf[1,2]):.4e} & {float(error_l1[2,2]):.4e} & {float(error_l2[2,2]):.4e} & {float(error_linf[2,2]):.4e} \\
            #         \midrule
            #         & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC \\
            #         \midrule
            #         \( 25 \)   & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} \\
            #         \( 50 \)   & {float(EOC_m_L1[0,1]):.3f} & {float(EOC_m_L2[0,1]):.3f} & {float(EOC_m_Linfty[0,1]):.3f} & {float(EOC_m_L1[1,1]):.3f} & {float(EOC_m_L2[1,1]):.3f} & {float(EOC_m_Linfty[1,1]):.3f} & {float(EOC_m_L1[2,1]):.3f} & {float(EOC_m_L2[2,1]):.3f} & {float(EOC_m_Linfty[2,1]):.3f} \\
            #         \( 100 \)  & {float(EOC_m_L1[0,2]):.3f} & {float(EOC_m_L2[0,2]):.3f} & {float(EOC_m_Linfty[0,2]):.3f} & {float(EOC_m_L1[1,2]):.3f} & {float(EOC_m_L2[1,2]):.3f} & {float(EOC_m_Linfty[1,2]):.3f} & {float(EOC_m_L1[2,2]):.3f} & {float(EOC_m_L2[2,2]):.3f} & {float(EOC_m_Linfty[2,2]):.3f} \\
            #         \bottomrule
            #     \end{{tabular}}
            # \end{{table}}'''
            print(latex_str, file=file)

            #print out time script took to run.
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time: {:.3f} seconds".format(elapsed_time),
                  file=file)


# def convergence_tables1D_2():

#     with open('convergence_tables1D.txt', 'w') as file:


#         r_gps = [1,2,3]
#         Nxs = np.array([25,50,100,200,400])

#         #function that assigns the grid data.
#         #exact_f = lambda x : np.sin(2*PI*x)

#         def shock(x):
#             y = np.zeros_like(x)

#             for i in range(len(x)):

#                 if x[i] < 1:
#                     y[i] = 0
#                 else:
#                     y[i] = 1
#             return y

#         high_precision = False

#         #EXACT SOLUTIONS
#         f = lambda x : np.sin(2*PI*x)
#         ddx = lambda x : np.cos(2*PI*x)* 2*PI
#         ddxx = lambda x : -np.sin(2*PI*x)* (2*PI)**2
#         ddxxx = lambda x : -np.cos(2*PI*x)* (2*PI)**3
#         integral = lambda x: -np.cos(2*PI*x)/(2*PI)
#         def volume_average(x):
#             f = lambda x : -np.cos(2*PI*x)/(2*PI)
#             dx = x[1] - x[0]
#             y = np.zeros_like(x)

#             for i in range(len(x)):
#                 y[i] = (f(x[i] + dx/2) - f(x[i] - dx/2))/dx
#             return y


#         exact_sols = [f, ddx, ddxx, ddxxx, volume_average, integral, f, f, shock, shock]
#         names = ["Pt_to_Pt", "Pt_to_Der", "Pt_to_2Der", "Pt_to_3der", "Pt_to_VolAve", "Pt_to_Int", "Pt_to_Pt_NN", "Pt_to_Pt_AS", "Pt_to_Pt_SE-SHOCK", "Pt_to_Pt_AS-SHOCK"]
#         dims = [1,1,1,1,1,1,1,1,1,1]

#         kint = sym.integrate(kern.SE, kern.x_sym)
#         left_integral = kint.subs(kern.x_sym, kern.x_sym-kern.dx_sym/2)
#         right_integral = kint.subs(kern.x_sym, kern.x_sym+kern.dx_sym/2)
#         T = (right_integral - left_integral)/kern.dx_sym

#         f1 = sym.integrate(kern.SE, kern.x_sym)
#         x_integral = (f1.subs(kern.x_sym, kern.x_sym + kern.dx_sym/2) - f1.subs(kern.x_sym, kern.x_sym - kern.dx_sym/2))/kern.dx_sym
#         f2 = sym.integrate(x_integral, kern.y_sym)
#         f2.simplify()
#         C = (f2.subs(kern.y_sym, kern.y_sym + kern.dx_sym/2) - f2.subs(kern.y_sym, kern.y_sym - kern.dx_sym/2))/kern.dx_sym
#         C = C.simplify()



#         k_kernels = [kern.SE, kern.SE, kern.SE, kern.SE, kern.SE, kern.SE, kern.AS2, kern.AS, kern.AS, kern.SE]
#         k_star_kernels = [kern.SE, sym.diff(kern.SE, kern.y_sym,1), sym.diff(kern.SE, kern.y_sym,2),
#                         sym.diff(kern.SE, kern.y_sym,3), T, C, kern.AS2, kern.AS, kern.AS, kern.SE]

#         assert(len(names) == len(exact_sols))
#         assert(len(names) == len(k_kernels))
#         assert(len(names) == len(k_star_kernels))
#         assert(len(names) == len(dims))

#         #for k in range(len(names)-2): #loop over each type of conversion.
#         for k in range(1):
#             print(names[k])

#             n_rgps = 3
#             n_Nxs = 6

#             error_l1 = np.zeros((n_rgps, n_Nxs))
#             error_l2 = np.zeros((n_rgps, n_Nxs))
#             error_linf =np.zeros((n_rgps, n_Nxs))
#             EOC_m_L1 = np.zeros((n_rgps, n_Nxs))
#             EOC_m_L2 = np.zeros((n_rgps, n_Nxs))
#             EOC_m_Linfty = np.zeros((n_rgps, n_Nxs))


#             fig, (ax1) = plt.subplots(1, 1, sharey=True)

#             for i, r_gp in enumerate(r_gps):
#                 for j, Nx in enumerate(Nxs):

#                     xlim = (0, 2*np.pi)


#                     g = Grid1D(xlim, Nx, r_gp)
#                     g.fill_grid(f)
#                     dx = g.x[1] - g.x[0]

#                     #keep ell/dx constant.
#                     #ell = 1.2 for Nx = 25 works well.
#                     # ell = (1.2/((xlim[1]-xlim[0])/25))*dx
#                     # print("Nx: ", Nx, " ell: ", ell)
#                     #ell = 0.8
#                     ell = 24*dx


#                     gprecipe = GP_recipe1D(g, r_gp, ell=ell, stencil_method ="center", high_precision=True)


#                     x_predict = g.x_int - dx/2
#                     x_predict = x_predict[1:]

#                     y_predict = gprecipe.convert_custom(x_predict, k_kernels[k], k_star_kernels[k])


#                     y_exact = exact_sols[k](x_predict)
#                     #print(type(y_exact))
#                     #mp_matrix = mm.matrix(y_exact.tolist())


#                     #L1 ERROR
#                     #err = np.linalg.norm(y_exact-  y_predict, ord=1)
#                     err = error.L1_error(y_exact, y_predict, dx)

#                     if high_precision:
#                         error_l1[i,j] = err
#                     else:
#                         error_l1[i][j] = err

#                     #L2 ERROR
#                     #err = np.linalg.norm(y_exact-  y_predict, ord=2)
#                     err = error.L2_error(y_exact, y_predict, dx)

#                     if high_precision:
#                         error_l2[i,j] = err
#                     else:
#                         error_l2[i][j] = err


#                     #LInfinity ERROR
#                     err = error.L_infinity_error(y_exact, y_predict)
#                     if high_precision:
#                         error_linf[i,j] = err
#                     else:
#                         error_linf[i][j] = err


#                     #plotting for debugging...
#                     # plt.figure()
#                     # plt.plot(g.x, g.grid, label="Training Function")
#                     # plt.plot(x_predict, y_predict, '.',label="GP sol")
#                     # plt.plot(x_predict, y_exact, '.', label="Exact Sol")
#                     # plt.title(f"{names[k]}  R_GP = {r_gp}")
#                     # plt.legend()
#                     # plt.show()


#             #Print Table.
#             print("========={}==============".format(names[k]), file=file)



#             for i, r_gp in enumerate(r_gps):
#                 print("----r_gp = {}----".format(r_gp), file=file)

#                 for j in range(len(Nxs)):

#                     if j == 0: #first pass, we don't have EOC.
#                         if high_precision:
#                             print(f"Nx: {Nxs[j]}  L1:  {float(error_l1[i,j]):.4e}   EOC: ----    L2: {float(error_l2[i,j]):.4e}   EOC: ---      L_INF: {float(error_linf[i,j]):.4e}   EOC: ---   ", file=file)
#                         else:
#                             print(f"Nx: {Nxs[j]}  L1:  {error_l1[i][j]:.4e}   EOC: ----    L2: {error_l2[i][j]:.4e}   EOC: ---", file=file)
#                     else:
#                         if high_precision:
#                             EOC_L1 = mm.log(error_l1[i,j-1]/ error_l1[i,j])/mm.log(2)
#                             EOC_L2 = mm.log(error_l2[i,j-1]/ error_l2[i,j])/mm.log(2)
#                             EOC_Linf = mm.log(error_linf[i,j-1]/ error_linf[i,j])/mm.log(2)


#                             print("EOC_L1: ", EOC_L1)
#                             print("EOC_L2: ", EOC_L2)

#                             print("EOC_Linf: ", EOC_Linf)

#                             # print(float(error_l1[i,j]))
#                             # print(EOC_L1)
#                             #try:
#                             #error_l1[i,j]
#                             print(f"Nx: {Nxs[j]}  L1:  {float(error_l1[i,j]):.4e}   EOC: {float(EOC_L1):.3f}    L2: {float(error_l2[i,j]):.4e}   EOC: {float(EOC_L2):.3f}      L_INF: {float(error_linf[i,j]):.4e}   EOC: {float(EOC_Linf):.3f}   ", file=file)
#                             # except:
#                             #     print(f"Nx: {Nxs[j]}  L1:  {error_l1[i,j]}   EOC: {EOC_L1)}    L2: {error_l2[i,j]):.4e}   EOC: {float(EOC_L2):.3f}      L_INF: {float(error_linf[i,j]):.4e}   EOC: {float(EOC_Linf):.3f}   ", file=file)

#                             EOC_m_L1[i,j] = EOC_L1
#                             EOC_m_L2[i,j] = EOC_L2
#                             EOC_m_Linfty[i,j] = EOC_Linf

#                         else:
#                             EOC_L1 = np.log(error_l1[i][j-1]/ error_l1[i][j])/np.log(2)
#                             EOC_L2 = np.log(error_l2[i][j-1]/ error_l2[i][j])/np.log(2)
#                             EOC_Linf = np.log(error_linf[i][j-1]/ error_linf[i][j])/np.log(2)
#                             EOC_m_L1[i][j] = EOC_L1
#                             EOC_m_L2[i][j] = EOC_L2
#                             EOC_m_Linfty[i][j] = EOC_Linf

#                             print(f"Nx: {Nxs[j]}  L1:  {error_l1[i][j]:.4e}   EOC: {EOC_L1:.3f}    L2: {error_l2[i][j]:.4e}   EOC: {EOC_L2:.3f}    L_INF: {error_linf[i][j]:.4e}   EOC: {EOC_Linf:.3f}", file=file)



#                             print("====")
#                             print("EOC_L1 setting: ", EOC_m_L1[i][j], " for i=", i, " j=",j)
#                             print("EOC_L1 setting: ", EOC_L1, " for i=", i, " j=",j)
#                             print("error_l1[i][j-1]: ", error_l1[i][j-1])
#                             print("error_l1[i][j]: ", error_l1[i][j])
#                             print("error_l1[i][j-1]/ error_l1[i][j]: ", error_l1[i][j-1]/ error_l1[i][j])
#                             EOC_L1 = np.log(error_l1[i][j-1]/ error_l1[i][j])/np.log(2)
#                             print("EOC_L1:", EOC_L1)


#             print("===latex===", file=file)
#             latex_str = rf'''\begin{{table}}[ht!]
#                     \footnotesize
#                     \centering
#                     \caption{{\text{names[k]}}}
#                     \label{{table:Pt_to_Pt_1D}}
#                     \begin{{tabular}}{{@{{}}cccc|ccc|cccc@{{}}}}
#                         \toprule
#                         \multirow{{2}}{{*}}{{Grid Res.}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 1$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 2$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 3$}} \\
#                         \cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}} \cmidrule(lr){{8-10}}
#                         & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error \\
#                         \midrule
#                         \( 25 \)   & {float(error_l1[0,0]):.4e} & {float(error_l2[0,0]):.4e} & {float(error_linf[0,0]):.4e} & {float(error_l1[1,0]):.4e} & {float(error_l2[1,0]):.4e} & {float(error_linf[1,0]):.4e} & {float(error_l1[2,0]):.4e} & {float(error_l2[2,0]):.4e} & {float(error_linf[2,0]):.4e} \\
#                         \( 50 \)   & {float(error_l1[0,1]):.4e} & {float(error_l2[0,1]):.4e} & {float(error_linf[0,1]):.4e} & {float(error_l1[1,1]):.4e} & {float(error_l2[1,1]):.4e} & {float(error_linf[1,1]):.4e} & {float(error_l1[2,1]):.4e} & {float(error_l2[2,1]):.4e} & {float(error_linf[2,1]):.4e} \\
#                         \( 100 \)  & {float(error_l1[0,2]):.4e} & {float(error_l2[0,2]):.4e} & {float(error_linf[0,2]):.4e} & {float(error_l1[1,2]):.4e} & {float(error_l2[1,2]):.4e} & {float(error_linf[1,2]):.4e} & {float(error_l1[2,2]):.4e} & {float(error_l2[2,2]):.4e} & {float(error_linf[2,2]):.4e} \\
#                         \( 200 \)  & {float(error_l1[0,3]):.4e} & {float(error_l2[0,3]):.4e} & {float(error_linf[0,3]):.4e} & {float(error_l1[1,3]):.4e} & {float(error_l2[1,3]):.4e} & {float(error_linf[1,3]):.4e} & {float(error_l1[2,3]):.4e} & {float(error_l2[2,3]):.4e} & {float(error_linf[2,3]):.4e} \\
#                         \( 400 \)  & {float(error_l1[0,4]):.4e} & {float(error_l2[0,4]):.4e} & {float(error_linf[0,4]):.4e} & {float(error_l1[1,4]):.4e} & {float(error_l2[1,4]):.4e} & {float(error_linf[1,4]):.4e} & {float(error_l1[2,4]):.4e} & {float(error_l2[2,4]):.4e} & {float(error_linf[2,4]):.4e} \\
#                         \( 800 \)  & {float(error_l1[0,5]):.4e} & {float(error_l2[0,5]):.4e} & {float(error_linf[0,5]):.4e} & {float(error_l1[1,5]):.4e} & {float(error_l2[1,5]):.4e} & {float(error_linf[1,5]):.4e} & {float(error_l1[2,5]):.4e} & {float(error_l2[2,5]):.4e} & {float(error_linf[2,5]):.4e} \\
#                         \midrule
#                         & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC \\
#                         \midrule
#                         \( 25 \)   & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} \\
#                         \( 50 \)   & {float(EOC_m_L1[0,1]):.3f} & {float(EOC_m_L2[0,1]):.3f} & {float(EOC_m_Linfty[0,1]):.3f} & {float(EOC_m_L1[1,1]):.3f} & {float(EOC_m_L2[1,1]):.3f} & {float(EOC_m_Linfty[1,1]):.3f} & {float(EOC_m_L1[2,1]):.3f} & {float(EOC_m_L2[2,1]):.3f} & {float(EOC_m_Linfty[2,1]):.3f} \\
#                         \( 100 \)  & {float(EOC_m_L1[0,2]):.3f} & {float(EOC_m_L2[0,2]):.3f} & {float(EOC_m_Linfty[0,2]):.3f} & {float(EOC_m_L1[1,2]):.3f} & {float(EOC_m_L2[1,2]):.3f} & {float(EOC_m_Linfty[1,2]):.3f} & {float(EOC_m_L1[2,2]):.3f} & {float(EOC_m_L2[2,2]):.3f} & {float(EOC_m_Linfty[2,2]):.3f} \\
#                         \( 200 \)  & {float(EOC_m_L1[0,3]):.3f} & {float(EOC_m_L2[0,3]):.3f} & {float(EOC_m_Linfty[0,3]):.3f} & {float(EOC_m_L1[1,3]):.3f} & {float(EOC_m_L2[1,3]):.3f} & {float(EOC_m_Linfty[1,3]):.3f} & {float(EOC_m_L1[2,3]):.3f} & {float(EOC_m_L2[2,3]):.3f} & {float(EOC_m_Linfty[2,3]):.3f} \\
#                         \( 400 \)  & {float(EOC_m_L1[0,4]):.3f} & {float(EOC_m_L2[0,4]):.3f} & {float(EOC_m_Linfty[0,4]):.3f} & {float(EOC_m_L1[1,4]):.3f} & {float(EOC_m_L2[1,4]):.3f} & {float(EOC_m_Linfty[1,4]):.3f} & {float(EOC_m_L1[2,4]):.3f} & {float(EOC_m_L2[2,4]):.3f} & {float(EOC_m_Linfty[2,4]):.3f} \\
#                         \( 800 \)  & {float(EOC_m_L1[0,5]):.3f} & {float(EOC_m_L2[0,5]):.3f} & {float(EOC_m_Linfty[0,5]):.3f} & {float(EOC_m_L1[1,5]):.3f} & {float(EOC_m_L2[1,5]):.3f} & {float(EOC_m_Linfty[1,5]):.3f} & {float(EOC_m_L1[2,5]):.3f} & {float(EOC_m_L2[2,5]):.3f} & {float(EOC_m_Linfty[2,5]):.3f} \\

#                         \bottomrule
#                     \end{{tabular}}
#                 \end{{table}}'''

#             # latex_str = rf'''\begin{{table}}[ht!]
#             #     \footnotesize
#             #     \centering
#             #     \caption{{names[k]}}
#             #     \label{{table:Pt_to_Pt_1D}}
#             #     \begin{{tabular}}{{@{{}}cccc|ccc|cccc@{{}}}}
#             #         \toprule
#             #         \multirow{{2}}{{*}}{{Grid Res.}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 1$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 2$}} & \multicolumn{{3}}{{c}}{{$R_{{gp}} = 3$}} \\
#             #         \cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}} \cmidrule(lr){{8-10}}
#             #         & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error & $L_1$ error &  $L_2$ error & $L_\infty$ error \\
#             #         \midrule
#             #         \( 25 \)   & {float(error_l1[0,0]):.4e} & {float(error_l2[0,0]):.4e} & {float(error_linf[0,0]):.4e} & {float(error_l1[1,0]):.4e} & {float(error_l2[1,0]):.4e} & {float(error_linf[1,0]):.4e} & {float(error_l1[2,0]):.4e} & {float(error_l2[2,0]):.4e} & {float(error_linf[2,0]):.4e} \\
#             #         \( 50 \)   & {float(error_l1[0,1]):.4e} & {float(error_l2[0,1]):.4e} & {float(error_linf[0,1]):.4e} & {float(error_l1[1,1]):.4e} & {float(error_l2[1,1]):.4e} & {float(error_linf[1,1]):.4e} & {float(error_l1[2,1]):.4e} & {float(error_l2[2,1]):.4e} & {float(error_linf[2,1]):.4e} \\
#             #         \( 100 \)  & {float(error_l1[0,2]):.4e} & {float(error_l2[0,2]):.4e} & {float(error_linf[0,2]):.4e} & {float(error_l1[1,2]):.4e} & {float(error_l2[1,2]):.4e} & {float(error_linf[1,2]):.4e} & {float(error_l1[2,2]):.4e} & {float(error_l2[2,2]):.4e} & {float(error_linf[2,2]):.4e} \\
#             #         \midrule
#             #         & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC & $L_1$ EOC &  $L_2$ EOC & $L_\infty$ EOC \\
#             #         \midrule
#             #         \( 25 \)   & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} & {"--"} \\
#             #         \( 50 \)   & {float(EOC_m_L1[0,1]):.3f} & {float(EOC_m_L2[0,1]):.3f} & {float(EOC_m_Linfty[0,1]):.3f} & {float(EOC_m_L1[1,1]):.3f} & {float(EOC_m_L2[1,1]):.3f} & {float(EOC_m_Linfty[1,1]):.3f} & {float(EOC_m_L1[2,1]):.3f} & {float(EOC_m_L2[2,1]):.3f} & {float(EOC_m_Linfty[2,1]):.3f} \\
#             #         \( 100 \)  & {float(EOC_m_L1[0,2]):.3f} & {float(EOC_m_L2[0,2]):.3f} & {float(EOC_m_Linfty[0,2]):.3f} & {float(EOC_m_L1[1,2]):.3f} & {float(EOC_m_L2[1,2]):.3f} & {float(EOC_m_Linfty[1,2]):.3f} & {float(EOC_m_L1[2,2]):.3f} & {float(EOC_m_L2[2,2]):.3f} & {float(EOC_m_Linfty[2,2]):.3f} \\
#             #         \bottomrule
#             #     \end{{tabular}}
#             # \end{{table}}'''
#             print(latex_str, file=file)





#             #print out time script took to run.
#             end_time = time.time()
#             elapsed_time = end_time - start_time
#             print("Elapsed time: {:.3f} seconds".format(elapsed_time), file=file)

if __name__ == "__main__":
    convergence_tables1D()
