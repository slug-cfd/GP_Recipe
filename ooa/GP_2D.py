from sympy import symbols, factorial, Function, simplify, pprint, exp, collect
import numpy as np
from ooa_kernel import SympyKernel2D, SE_2d, x_sym1, y_sym1, x_sym2, y_sym2
import sys
from stencil_helper_2D import get_points
import sympy as sym


def taylor_expansion_2D(n_terms, h1, h2, expand_point1, expand_point2):
    """
    Creates a 2D Taylor expansion with n terms
    h1, h2: the symbols for x and y directions
    expand_point1, expand_point2: points to expand around in x and y
    """
    f = Function('f')
    expansion = 0

    # Double sum for 2D Taylor expansions
    for i in range(n_terms):
        for j in range(n_terms-i):  # This ensures total order is less than n_terms
            # f_x{i}_y{j} represents ∂^i+j f/∂x^i ∂y^j evaluated at (expand_point1, expand_point2)
            coeff = symbols(f'f_x{i}_y{j}')
            term = coeff * (h1 - expand_point1)**i * (h2 - expand_point2)**j / (factorial(i) * factorial(j))
            expansion += term
    return expansion


# a main driver routine
def run(conversion, r_gp, xstarLoc, ystarLoc, h, ell, stencil, verbose = False):


    # how many terms we want in 2D Taylor series expansions
    numTaylorSeriesTerms = 12

    if conversion == 0:
        k_star_kernel = SE_2d
        if xstarLoc == 0 and ystarLoc == 0:
            print("Not doing calcuation, requested point to point at cell center (0,0). This is an exact calculation.")
            return -99
    elif conversion == 1:
        k_star_kernel = sym.diff(SE_2d, x_sym1) #symbolic dx
    elif conversion == 2:
        k_star_kernel = sym.diff(SE_2d, y_sym1) #symbolic dy
    elif conversion == 3:
        k_star_kernel = sym.diff(sym.diff(SE_2d, x_sym1), x_sym1) #symbolic dxdx
    elif conversion == 4:
        k_star_kernel = sym.diff(sym.diff(SE_2d, x_sym1), y_sym1) #symbolic dxdy
    elif conversion == 5:
        k_star_kernel = sym.diff(sym.diff(SE_2d, y_sym1), y_sym1) #symbolic dydy


    # Set GP kernels K and k_*
    kernel        = SympyKernel2D(SE_2d, ell = ell) # Kernel used in K matrix
    k_star_kernel = SympyKernel2D(k_star_kernel, ell = ell) #kernel used in k_star vector



    # Define all points in the stencil
    points = get_points(h, stencil, r_gp)
    # print("points:")
    # print(points)



    # Compute kernel matrix numerically
    n_points = len(points)
    K = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            K[i,j] = kernel(points[i], points[j], h, h)
    # Compute k_star vector
    k_star = np.zeros(n_points)
    x_star = xstarLoc*h
    y_star = ystarLoc*h



    for i in range(n_points):
        k_star[i] = k_star_kernel(points[i], (x_star, y_star), h, h)

    for i in range(n_points):
        x1,y1 = points[i]
        x2,y2 = (x_star, y_star)


    #multiply numerically by h, so we can symbolically divide by h later

    if conversion == 0: #der order = 0
        weights = np.matmul(k_star, np.linalg.inv(K))
    elif conversion == 1 or conversion == 2: #der order = 1
        weights = np.matmul(k_star, np.linalg.inv(K))*h
    elif conversion == 3 or conversion == 4 or conversion == 5: #der order = 2
        weights = np.matmul(k_star, np.linalg.inv(K))*h**2


    if np.all(np.abs(weights) < 1e-13): 
        print("All weights are 0. This should converge to 0th order")
        return 0




    dx = symbols('h')

    #this works with symbolic points too :)
    #ah life is good
    symbolic_points = get_points(dx, stencil, r_gp)



    fs = []
    for point in symbolic_points:
        f_xh_yh = taylor_expansion_2D(numTaylorSeriesTerms, point[0], point[1], xstarLoc*dx, ystarLoc*dx)
        fs.append(f_xh_yh)



    pred = np.dot(weights, fs)



    if conversion == 0: #der order = 0
        pred = simplify(pred) #symbolic division by h
    elif conversion == 1 or conversion == 2: #der order = 1
        pred = simplify(pred/dx) #symbolic division by h
    elif conversion == 3 or conversion == 4 or conversion == 5: #der order = 2
        pred = simplify(pred/dx/dx) #symbolic division by h




    # Collect terms with respect to dx (h)
    pred_coll = collect(pred, dx)
    if verbose:
        print('== GP prediction in Taylor expansion ==')
        print(pred_coll)
        print()
        print()

    # Define main_term_order based on conversion type
    if conversion == 0:  # pt to pt
        main_term_order = 0
    elif conversion in [1, 2]:  # pt to dx or dy
        main_term_order = 0
    elif conversion in [3, 4, 5]:  # pt to dxdx, dxdy, or dydy
        main_term_order = 0

    # Extract coefficients more directly from the expression
    coefficients = {}
    max_power = numTaylorSeriesTerms

    # Handle negative powers and constant terms
    for power in range(-2, max_power):
        term = pred.coeff(dx, power)
        if term != 0:
            # Extract coefficients for each f_xi_yj term
            for i in range(numTaylorSeriesTerms):
                for j in range(numTaylorSeriesTerms):
                    symbol = symbols(f'f_x{i}_y{j}')
                    coeff = term.coeff(symbol)
                    if coeff != 0:
                        coefficients[(i, j, power)] = float(coeff)

    if verbose:
        for (i, j, h_order), coeff in coefficients.items():
            print(f"h^{h_order}: f_x{i}_y{j} = {coeff}")

    # Rest of the code remains the same
    non_main_coeffs = []
    for total_order in range(max_power):
        order_max_coeff = 0
        if verbose:
            print(f"\nChecking order {total_order}:")
        for (i, j, h_order), coeff in coefficients.items():
            if h_order == total_order and total_order > main_term_order:
                if verbose:
                    print(f"Found coefficient: {coeff} for f_x{i}_y{j}")
                order_max_coeff = max(order_max_coeff, abs(float(coeff)))
        if order_max_coeff > 0:
            if verbose:
                print(f"Max coeff for order {total_order}: {order_max_coeff}")
            non_main_coeffs.append((order_max_coeff, total_order))


    if non_main_coeffs:
        max_coeff, gp_order = max(non_main_coeffs)
        if verbose:
            print('\n The largest coefficient in magnitude = ', 
                np.format_float_scientific(max_coeff, unique=False, precision=8))
            
    # Add effective order calculation
    numerator = 0
    denominator = 0
    
    # Only consider terms with positive powers of h (order > main_term_order)
    for (i, j, h_order), coeff in coefficients.items():
        if h_order > main_term_order:  # Only consider error terms
            numerator += h_order * abs(coeff)
            denominator += abs(coeff)
    
    p_eff = numerator / denominator if denominator != 0 else 0
    
    conversion_dict = {0: "pt to pt", 1: "pt to dx", 2: "pt to dy", 3: "pt to dxdx", 4: "pt to dxdy", 5: "pt to dydy"}

    print("r_gp = ", r_gp)
    print("starloc = (", xstarLoc, ",", ystarLoc, ")")
    print("conversion = ", conversion_dict[conversion])
    print("stencil = ", stencil)
    print('Expected GP order = ', gp_order)
    print('Effective GP order of convergence (p_eff) = {:.8f}'.format(p_eff))

    return p_eff






if __name__ == "__main__":
    '''
    # Set numerical values
    h = 0.005      # grid spacing
    ell=6*h
    # Star point (query point)

    ##########################################
    ######     Choose Inputs           ######
    ##########################################
    #[INPUT 1] Choose r_gp
    # Set r_gp (r_gp = 1, 2, or 3)
    r_gp = 1
    # r_gp = 2
    # r_gp = 3


    #[INPUT 2] Choose xstar query location
    # (a) xstarLoc = 0.0 for central point at x=x_i or
    # (b) xstarLoc = 0.5 for interface location x=x_{i+1/2}
    xstarLoc = 0.5
    x_star = xstarLoc*h

    #[INPUT 3] Choose ystar query location
    # (a) ystarLoc = 0.0 for central point at y=y_i or
    # (b) ystarLoc = 0.5 for interface location y=y_{i+1/2}
    ystarLoc = 0.5
    y_star = ystarLoc*h

    starLoc = (xstarLoc, ystarLoc)


    #[INPUT 4] Choose Conversion
    #conversion = 0#pt to pt
    conversion = 1#pt to dx
    # conversion = 2#pt to dy
    # conversion = 3#pt to dxdx
    # conversion = 4#pt to dxdy
    # conversion = 5#pt to dydy



    #[INPUT 5] Choose Stencil
    stencil = "square" #diamond, or blocky_diamond
    ##########################################
    ######     End of Inputs      ######
    ##########################################
    run(conversion, r_gp, starLoc[0], starLoc[1], h, ell, stencil, verbose = False)
    '''



    #TABLE 1!
    stencils = ["square", "blocky_diamond", "diamond", "cross"]
    conversions = [0]
    r_gps = [2]
    h = 0.005
    ell = 6*h
    starLocs = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]

    for stencil in stencils:
        for conversion in conversions:
            for xstarLoc in starLocs:
                for r_gp in r_gps:

                    print("==========:1==========")
                    #run(conversion, r_gp, xstarLoc[0], xstarLoc[1], h, ell, stencil, verbose=False)


    #TABLE 2!
    stencils = ["cross"]

    for stencil in stencils:
        conversions = [0,1,2,3,4,5]
        r_gps = [2]
        h = 0.005
        ell = 6*h
        starLocs = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]

        solutionsGrid = np.zeros((len(conversions), len(starLocs)))

        for i,conversion in enumerate(conversions):
            for j,xstarLoc in enumerate(starLocs):
                for r_gp in r_gps:
                    print("==========:2==========")
                    p_eff = run(conversion, r_gp, xstarLoc[0], xstarLoc[1], h, ell, stencil, verbose=False)
                    solutionsGrid[i,j] = p_eff


        #latex table header
        latex_table = r"""
        \begin{table}[h!]
            \centering
            \begin{tabular}{|c|c|c|c|}
                \hline
                \textbf{Conversion} & $p_{\text{eff}} \text{ to } x_* = (0, 0)$ & $p_{\text{eff}} \text{ to } x_* = (\frac{1}{2}, 0)$ & $p_{\text{eff}} \text{ to } x_* = (\frac{1}{2}, \frac{1}{2})$\\
                \hline
        """

        # Conversion names
        conversion_names = [
            "pt to pt",
            "pt to dx",
            "pt to dy",
            "pt to dxdx",
            "pt to dxdy",
            "pt to dydy"
        ]

        #table values
        for i, conversion in enumerate(conversion_names):
            latex_table += f"        {conversion} & {solutionsGrid[i, 0]:.4f} & {solutionsGrid[i, 1]:.4f} & {solutionsGrid[i, 2]:.4f} \\\\\n"

        #close the table
        latex_table += r"""        \hline
            \end{tabular}
            \caption{Expected and effective GP convergence orders for different derivative approximations using the """ + stencil + r""" stencil with $r_{gp}=2$.}
            \label{tab:""" + stencil + r"""_convergence}
        \end{table}
        """



        print(latex_table)

