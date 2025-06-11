import sympy as sym
import numpy as np

class SympyKernel2D:
    def __init__(self, kernel, **kwargs):
        l_val = kwargs.get("ell", 0.8)
        nu_val = kwargs.get("nu", 1.5)
        high_precision = kwargs.get("high_precision", False)
        alpha_val = kwargs.get("alpha", 1.0)
        beta_val = kwargs.get("beta", 0.0)
        sigma_val = kwargs.get("sigma", 1.0)
        sigma0_val = kwargs.get("sigma0", 1.0)

        # Create symbols
        x_sym1, x_sym2, y_sym1, y_sym2, dx_sym, dy_sym, l_sym, nu_sym, alpha_sym, beta_sym, sigma_sym, sigma0_sym = sym.symbols('x1 x2 y1 y2 dx dy l nu alpha beta sigma sigma0')

        # Substitute kernel's parameters
        kernel_sub = kernel.subs({l_sym: l_val, nu_sym: nu_val, alpha_sym: alpha_val, beta_sym: beta_val, sigma_sym : sigma_val, sigma0_sym : sigma0_val}).evalf(n=500)

        # Create a lambda function from the sympy expression
        if high_precision:
            self.kernel = sym.lambdify((x_sym1, x_sym2, y_sym1, y_sym2, dx_sym, dy_sym), kernel_sub, 'mpmath')
        else:
            self.kernel = sym.lambdify((x_sym1, x_sym2, y_sym1, y_sym2, dx_sym, dy_sym), kernel_sub, 'numpy')

    def __call__(self, point1, point2, dx_val, dy_val, **kwargs):
        assert(len(point1) == 2)
        assert(len(point2) == 2)
        
        # Call the lambda function with the given points and deltas
        k = self.kernel(point1[0], point2[0], point1[1], point2[1], dx_val, dy_val)

        # Handle numpy numbers
        if isinstance(k, np.number):
            return k.item()
        else:
            return k
#2D variables
x_sym1 = sym.symbols('x1')
y_sym1 = sym.symbols('y1')
x_sym2 = sym.symbols('x2')
y_sym2 = sym.symbols('y2')
dx_sym = sym.symbols('dx')
dy_sym = sym.symbols('dy')

l_sym = sym.symbols('l')


#Euclidian Distance in 2D
d_2D = sym.sqrt((x_sym1-x_sym2)**2 + (y_sym1-y_sym2)**2)

SE_2d = sym.exp(-(x_sym1-x_sym2)**2/(2*l_sym**2)) * sym.exp(-(y_sym1-y_sym2)**2/(2*l_sym**2))




class SympyKernel1D:
    def __init__(self, kernel, **kwargs):
        l_val = kwargs.get("ell", 0.8)
        nu_val = kwargs.get("nu", 1.5)
        high_precision = kwargs.get("high_precision", False)
        alpha_val = kwargs.get("alpha", 1.0)
        beta_val = kwargs.get("beta", 0.0)
        sigma_val = kwargs.get("sigma", 1.0)
        sigma0_val = kwargs.get("sigma0", 1.0)

        # Create symbols
        x_sym, y_sym, dx_sym, l_sym, nu_sym, alpha_sym, beta_sym, sigma_sym, sigma0_sym = sym.symbols('x y dx l nu alpha beta sigma sigma0')

        # Substitute kernel's parameters
        kernel_sub = kernel.subs({l_sym: l_val, nu_sym: nu_val, alpha_sym: alpha_val, beta_sym: beta_val, sigma_sym : sigma_val, sigma0_sym : sigma0_val})

        # Create a lambda function from the sympy expression

        if high_precision:
            self.kernel = sym.lambdify((x_sym, y_sym, dx_sym), kernel_sub, 'mpmath')
        else:
            self.kernel = sym.lambdify((x_sym, y_sym, dx_sym), kernel_sub, 'numpy')
    def __call__(self, x_val, y_val, dx_val, **kwargs):
        # Call the lambda function with x_val, y_val, dx_val
        #keep **kwargs for backwards compatibablity with the old version.
        k = self.kernel(x_val, y_val, dx_val)

        # # Handle numpy numbers
        if isinstance(k, np.number):
            return k.item()
        else:
            return k


x_sym = sym.symbols('x')
y_sym = sym.symbols('y')
l_sym = sym.symbols('l')
l_sym = sym.symbols('l')
sigma0_sym = sym.symbols('sigma0')
sigma_sym = sym.symbols('sigma')


SE = sym.exp(-(x_sym-y_sym)**2/(2*l_sym**2)) 

d = sym.sqrt(( x_sym-y_sym )**2)
inner_prod = (sym.exp(-d)) /  (1 + 2*(sigma0_sym**2 + sigma_sym**2 * (x_sym-y_sym)**2))
DAS_V9 = 2 / sym.pi * sym.asin(inner_prod )