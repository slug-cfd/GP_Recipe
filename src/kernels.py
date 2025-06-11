import mymath as math
import numpy as np
import sympy as sym
import mpmath as mm

#KERNELS SHOULD BE EVAULATED OF THE FORM:
#K_SE(x,y,dx,l, high_precision)

#kernel variables
x_sym = sym.symbols('x')
y_sym = sym.symbols('y')
l_sym = sym.symbols('l')
dx_sym = sym.symbols('dx')
nu_sym = sym.symbols('nu')
alpha_sym = sym.symbols('alpha')
beta_sym = sym.symbols('beta')
sigma_sym = sym.symbols('sigma')
sigma0_sym = sym.symbols('sigma0')

#2D variables
x_sym1 = sym.symbols('x1')
y_sym1 = sym.symbols('y1')
x_sym2 = sym.symbols('x2')
y_sym2 = sym.symbols('y2')
dy_sym = sym.symbols('dy')


#Euclidian Distance in 1D
d = sym.sqrt((x_sym-y_sym)**2)

#Euclidian Distance in 2D
d_2D = sym.sqrt((x_sym1-x_sym2)**2 + (y_sym1-y_sym2)**2)


#1D Squared Exponential Kernel
#SE = sym.exp(-(d)**2/(2*l_sym**2))
SE = sym.exp(-(x_sym-y_sym)**2/(2*l_sym**2))

# x_sym_2d = x_sym1-x_sym2
# y_sym_2d = y_sym1-y_sym2
SE_2d = sym.exp(-(x_sym1-x_sym2)**2/(2*l_sym**2)) * sym.exp(-(y_sym1-y_sym2)**2/(2*l_sym**2))
#1D Inverse quadratic kernel
IQ =  (1 + (x_sym-y_sym)**2 / (l_sym**2)) **-1

#1D Multi quadratic kernel
MQ =  sym.sqrt(1 + (x_sym-y_sym)**2 / (l_sym**2))

import sympy as sp


#eq 4.29 in GP Book "arcsine kernel"
#sigma = 1
#sigma0 = 1
# inner_prod = 2 * x_sym * y_sym / (sp.sqrt((1 + 2*x_sym**2) * (1 + 2*y_sym**2))) 
# AS2 = 2 / sym.pi * sym.asin(inner_prod)

inner_prod = 2 * (sigma0_sym**2 + sigma_sym**2 * x_sym * y_sym) / ( (sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * x_sym**2))) * sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * y_sym**2)))
AS2 = 2 / sym.pi * sym.asin(inner_prod)

#2D implementation of neural network kernel eq 4.29 same as above (AS2)
#2X^T Σ X'
top = 2 * (sigma0_sym**2 + sigma_sym**2*x_sym1*x_sym2 + sigma_sym**2*y_sym1*y_sym2)
#sqrt( (1+2X^T Σ X) (1+2X'^T Σ X'))
bottom = sym.sqrt(1 + 2*(sigma0_sym**2 + x_sym1*sigma_sym**2*x_sym1 + y_sym1*sigma_sym**2*y_sym1)) \
        * sym.sqrt(1 + 2*(sigma0_sym**2 + x_sym2*sigma_sym**2*x_sym2 + y_sym1*sigma_sym**2*y_sym2)) 
K_NN_2D = 2/sym.pi * sym.asin(top/bottom)




#d = sqrt((x-y)**2)
# inner_prod = (d) / (sp.sqrt((1 + 2*x_sym**2) * (1 + 2*y_sym**2)))
# AS = 2 / sym.pi * sym.asin(inner_prod)
# sigma0 = 1
# sigma = 1


#original d in DAS.
d = sym.sqrt(( x_sym-y_sym)**2)

inner_prod = (d) /  ( (sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * x_sym**2))) * sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * y_sym**2)))
AS = 2 / sym.pi * sym.asin(inner_prod)


#testing trying to get as to be SPD
d = sym.sqrt( sym.exp(-(x_sym-y_sym)**2))
inner_prod = (d) /  ( (sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * x_sym**2))) * sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * y_sym**2)))
DAS_v2 = 2 / sym.pi * sym.asin(inner_prod)

#d = sym.sqrt(sym.exp(-(x_sym-y_sym)**2))
d = (1 + 2*(sigma0_sym**2 + sigma_sym**2 * x_sym*y_sym)) * sym.exp(-(x_sym-y_sym)**2)
inner_prod = (d) /  ( (sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * x_sym**2))) * sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * y_sym**2)))
DAS_v3 = 2 / sym.pi * sym.asin(inner_prod**2)


d = sym.sqrt(( x_sym-y_sym)**2) + (sp.sqrt(x_sym**2) + sp.sqrt(y_sym**2))
inner_prod = (d) /  ( (sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * x_sym**2))) * sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * y_sym**2)))
DAS_V4 = 2 / sym.pi * sym.asin(inner_prod)

d = sym.sqrt(( x_sym-y_sym)**2) + (sp.sqrt(x_sym**2) + sp.sqrt(y_sym**2))
inner_prod = (d) /   (sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * (x_sym-y_sym)**2))) 
DAS_V5 = 2 / sym.pi * sym.asin(inner_prod + sym.KroneckerDelta(x_sym,y_sym))


d = sym.sqrt(( x_sym-y_sym)**2)
inner_prod = (d) /  ( (sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * x_sym**2))) * sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * y_sym**2)))
DAS_V6 = 2 / sym.pi * sym.asin(inner_prod +  0.1*sym.KroneckerDelta(x_sym,y_sym))

d = sym.sqrt(( x_sym-y_sym)**2)
inner_prod = (sym.exp(-d)) /  ( (sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * x_sym**2))) * sp.sqrt(1 + 2*(sigma0_sym**2 + sigma_sym**2 * y_sym**2)))
DAS_V8 = 2 / sym.pi * sym.asin(inner_prod )

d = sym.sqrt(( x_sym-y_sym )**2)
inner_prod = (sym.exp(-d)) /  (1 + 2*(sigma0_sym**2 + sigma_sym**2 * (x_sym-y_sym)**2))
DAS_V9 = 2 / sym.pi * sym.asin(inner_prod )

d_2D = sym.sqrt(( x_sym1-x_sym2 )**2 + (y_sym1 - y_sym2)**2)
inner_prod = (sym.exp(-d_2D)) / (1 + 2*(sigma0_sym**2 + sigma_sym**2 * (x_sym1-x_sym2)**2))* (1 + 2*(sigma0_sym**2 + sigma_sym**2 * (y_sym1-y_sym2)**2))
DAS_V9_2D = 2 / sym.pi * sym.asin(inner_prod )


#2D implementation of my AS!
#bottom is same from K_NN_2D - 

bottom = sym.sqrt(1 + 2*(sigma0_sym**2 + x_sym1*sigma_sym**2*x_sym1 + y_sym1*sigma_sym**2*y_sym1)) \
        * sym.sqrt(1 + 2*(sigma0_sym**2 + x_sym2*sigma_sym**2*x_sym2 + y_sym1*sigma_sym**2*y_sym2)) 

AS_2D = (2/sym.pi) * sym.asin(d_2D/bottom)



#eq 4.29 in GP Book "arcsine kernel" with the augmented input vector
inner_prod = 2 * (1 +x_sym * y_sym) / (sp.sqrt((1 + 2*(x_sym**2 + 1)) * (1 + 2*(y_sym**2+1)))) 
AS1 = 2 / sym.pi * sym.asin(inner_prod)

#d = sqrt((x-y)**2)
inner_prod = (d) / (sp.sqrt((1 + 2*x_sym**2 + 1) * (1 + 2*y_sym**2 + 1)))
AS3 = 2 / sym.pi * sym.asin(inner_prod)

inner_prod = (sym.exp(-(x_sym-y_sym)**2/(2*l_sym**2))) / (sp.sqrt((1 + 2*x_sym**2) * (1 + 2*y_sym**2))) 
AS4 = 2 / sym.pi * sym.asin(inner_prod)
#d = sqrt((x-y)**2)


#some trials of new kernels with dongwook:
# d = (x_sym-y_sym)**2
# inner_prod = (d) / (sp.sqrt((1 + 2*x_sym**2) * (1 + 2*y_sym**2)))
# AS4 = 2 / sym.pi * sym.asin(inner_prod)

#Use SE as numerator and AS as denomenator. 

lower =  (sp.sqrt((1 + 2*x_sym**2) * (1 + 2*y_sym**2))) 
inner_prod = (SE) / lower
#SE proeduces 1, lower produces result over 1. but for some reason, SE/lower is still over 1.
#i believe this is related to precision. So for now we'll just subtract eps() from arcsin so we 
#don't take the arcsin of a value of a value over 1... This worksl
SEAS = 2 / sym.pi * sym.asin(inner_prod - 0.0000000000001)


#hyperbolic tangent kernel
HT = sp.tanh(alpha_sym * d + beta_sym)

#weights to turn on/off the kernel.
#if d evaluates to zero, a is 1 and b is 0
#if d doesn't evaluate to 0, b is 0 and a is 1.
eq = sym.Eq(d, 0)
ne = sym.Ne(d,0)
a = sym.Piecewise((1, eq), (0, True))
b = sym.Piecewise((1, ne), (0, True))



Matern = b*(1/ (sym.gamma(nu_sym) * 2**(nu_sym-1))) * (sym.sqrt(2*nu_sym)/l_sym * d)**nu_sym * sym.besselk(nu_sym, sym.sqrt(2*(nu_sym))/l_sym * d) \
       + a*1

#1D matern kernel
# Matern = sym.Piecewise(
#     ( (lambda: np.finfo(float).eps), d == 0), #if distance is 0 set to machine epsilon. See: https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b7e21201cfffb118934999025fd50cca/sklearn/gaussian_process/kernels.py#L1709
#     ((1/ (sym.gamma(nu_sym) * 2**(nu_sym-1))) * (sym.sqrt(2*nu_sym)/l_sym * d)**nu_sym * sym.besselk(nu_sym, sym.sqrt(2*(nu_sym))/l_sym * d), True)
# )

# Matern = (1/ (sym.gamma(nu_sym) * 2**(nu_sym-1))) * (sym.sqrt(2*nu_sym)/l_sym * d)**nu_sym * sym.besselk(nu_sym, sym.sqrt(2*(nu_sym))/l_sym * d)

# #modifiefd bessel function of second kind. K_nu
# sympy.functions.special.bessel.besselk(nu, z)
# #Gamma function
# sympy.functions.special.gamma_functions.gamma(t)




#Sympy has a really hard time with 2D C and all the intergals. We'll hard code it, dimension by dimension here.
#This form was originally derived by Adam.


#d_2D = sym.sqrt((x_sym1-x_sym2)**2 + (y_sym1-y_sym2)**2)
erf = sym.erf
ospi  = 1/sym.sqrt(sym.pi)


ddx = (x_sym1 - x_sym2)/dx_sym
ddy = (y_sym1 - y_sym2)/dx_sym

lod = l_sym/dx_sym

a11 = ( (ddx+1) / (sym.sqrt(2)*lod) )
a12 = ( (ddx-1) / (sym.sqrt(2)*lod) )

a21 = - ( ( (ddx+1)**2) / (2*lod**2) )
a22 = - ( ( (ddx-1)**2) / (2*lod**2) )

a31 =   ddx     /(sym.sqrt(2)*lod    )
a32 = -(ddx**2) /(2. *lod**2)

r =    (sym.sqrt(sym.pi)*(lod)**2)*  (a11 * erf(a11) + a12 * erf(a12) + (ospi)   * (       sym.exp(a21) +       sym.exp(a22)) -    2 * ( a31 * erf(a31) + ospi* sym.exp(a32)))

a11 = ( (ddy+1) / (sym.sqrt(2)*lod) )
a12 = ( (ddy-1) / (sym.sqrt(2)*lod) )

a21 = - ( ( (ddy+1)**2) / (2*lod**2) )
a22 = - ( ( (ddy-1)**2) / (2*lod**2) )

a31 =   ddy     /(sym.sqrt(2)*lod    )
a32 = -(ddy**2) /(     2. *lod**2 )

#!! DL -- eqn (11) in (x-direction) x (y-direction)
#!eq 12 still
r = r * (sym.sqrt(sym.pi)*(lod)**2)* (a11 * erf(a11) + a12 * erf(a12)  + (ospi)   * (       sym.exp(a21) +       sym.exp(a22)) -    2 * ( a31 * erf(a31) + ospi* sym.exp(a32)))

C_2D = r


T_2Dx = sym.sqrt(sym.pi/2) * lod * erf( (x_sym1 - x_sym2 + 0.5) / (sym.sqrt(2)*lod)) - erf( (x_sym1 - x_sym2 -0.5) / (sym.sqrt(2)*lod)) 
T_2Dy = sym.sqrt(sym.pi/2) * lod * erf( (y_sym1 - y_sym2 + 0.5) / (sym.sqrt(2)*lod)) - erf( (y_sym1 - y_sym2 -0.5) / (sym.sqrt(2)*lod)) 
T_2D = T_2Dx* T_2Dy





def get_sympy_kernel_1D(kernel):

    def make_sympy_kernel(x_val, y_val, dx_val, **kwargs):
        l_val = kwargs.get("ell", 0.8)
        nu_val = kwargs.get("nu", 1.5)
        high_precision = kwargs.get("high_precision", False)
        alpha_val = kwargs.get("alpha", 1.0)
        beta_val = kwargs.get("beta", 0.0)





        #evaluate in two steps to not blow up. First evaluate d and if it's zero we can 
        #omit most of the calcualtion.
        k = kernel.subs({x_sym: x_val, y_sym: y_val}).evalf(n=50)
        k = k.subs({dx_sym: dx_val, l_sym : l_val, nu_sym: nu_val, alpha_sym: alpha_val, beta_sym : beta_val}).evalf(n=50)

        return k
        if isinstance(k, sym.Expr):

            if k.is_number:
                return k
            else:
                #is sympy expression still :( grab coefficient.
                print("still sympy expression after kernel evaluation :( ")
                assert(false)

        else:
            #is a number now.
            return k

    return make_sympy_kernel


#converting to class. This will allow us to store and save the sympy result, then lambidfy which will make this 
#much faster. Curretly over 95% of computation time is spent in this method.
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
def get_sympy_kernel_2D(kernel):

    def make_sympy_kernel(point1, point2, dx_val, dy_val, **kwargs):
        assert(len(point1) == 2)
        assert(len(point2) == 2)
        l_val = kwargs.get("ell", 0.8)
        nu_val = kwargs.get("nu", 1.5)
        high_precision = kwargs.get("high_precision", False)
        alpha_val = kwargs.get("alpha", 1.0)
        beta_val = kwargs.get("beta", 0.0)
        sigma_val = kwargs.get("sigma", 1.0)
        sigma0_val = kwargs.get("sigma0", 1.0)


        #evaluate in two steps to not blow up. First evaluate d and if it's zero we can 
        #omit most of the calcualtion.
        k = kernel.subs({x_sym1: point1[0], x_sym2: point2[0], y_sym1:point1[1], y_sym2:point2[1]}).evalf(n=50)
        k = k.subs({dx_sym: dx_val, dy_sym: dy_val, l_sym : l_val, nu_sym: nu_val, alpha_sym: alpha_val, beta_sym : beta_val}).evalf(n=50)

        return k
        if isinstance(k, sym.Expr):

            if k.is_number:
                return k
            else:
                #is sympy expression still :( grab coefficient.
                print("still sympy expression after kernel evaluation :( ")
                assert(false)

        else:
            #is a number now.
            return k

    return make_sympy_kernel

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


    
# #2D Squared Exponential Kernel
# x_0, x_1 = sym.symbols('x_0, x_1')
# x = [x_0, x_1]
# y_0, y_1 = sym.symbols('y_0, y_1')
# y = [y_0, y_1]
# l = sym.symbols('l')
# x = sym.Matrix(x)
# y = sym.Matrix(y)
# z = x-y
# z= -z.applyfunc(lambda x: x**2)
# SE_2D = sym.exp(sum(z)/l)

