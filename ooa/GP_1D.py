

#The bottom line is:
#(1) GP’s k-th order appears in the O(h^k) term whose coefficient is the biggest in magnitude among all.

#(2) Unlike polynomial methods, GP’s coefficients are never close to zero in general
# as they don't get canceled but rather they are all *linearly combined.*
# These non-zero coefficients in the Taylor expansion were what confused us.

#(3) The largest coeff term is the leading error term that determines the GP’s accuracy.



from sympy import * #symbols, factorial, Function, simplify, pprint, expand, collect, degree
from ooa_kernel import SympyKernel1D, SE, x_sym, y_sym, DAS_V9
import numpy as np
import sympy as sym
import sys

def taylor_expansion(n_terms, h, expand_point):
    f = Function('f')
    expansion = 0
    for i in range(n_terms):
        expansion += symbols(f'f{i}') * (h - expand_point)**i / factorial(i)
    return expansion



# Set numerical values
h = 0.1      # grid spacing
ell = 6.0*h
#k_kernel = SE
k_kernel = DAS_V9
dx = symbols('h')

##########################################
######  Choose Three Inputs       ########
##########################################
#[INPUT 0] Choose centered vs. biased GP stencil
# biased-stencil (2*r_gp) for pt to pt vs.
# centered-stencil: (2*r_gp + 1) for pt to pt
# DL -- 12/11/24:
# * biased stencil orders are (a bit weird?)
# (a) 2-2-2 for pt, dx, dxdx with r_gp=1
# (b) 4-4-2 for pt, dx, dxdx with r_gp=2
# (c) 6-6-4 for pt, dx, dxdx with r_gp=3
centered_gp = True



#[INPUT 1] Choose r_gp
# Set r_gp (r_gp = 1, 2, or 3)
# GP radius
#r_gp = 1
r_gp = 2
#r_gp = 3



#[INPUT 2] Choose query location
# (a) xstarLoc = 0.0 for central point at x=x_i or
# (b) xstarLoc = 0.5 for interface location x=x_{i+1/2}
xstarLoc = 0.5
x_star = xstarLoc*h



#[INPUT 3] Choose derivative order
der_ordr = 0 # pt to pt
#der_ordr = 1 # pt to 1st der
#der_ordr = 2 # pt to 2nd der
##########################################
######     End of Three Inputs      ######
##########################################




if der_ordr == 0:
    #1. pt to pt
    k_star_kernel = SE
elif der_ordr == 1:
    #2. pt to 1st der
    k_star_kernel = sym.diff(SE, y_sym) #symbolic dx
elif der_ordr == 2:
    #3. pt to 2nd der
    k_star_kernel = sym.diff(sym.diff(SE, x_sym), x_sym) #symbolic dxdx





kernel = SympyKernel1D(SE, ell=ell) # Kernel used in K matrix
k_star_kernel = SympyKernel1D(k_star_kernel, ell=ell) #kernel used in k_star vector


if der_ordr == 0 and xstarLoc  == 0:
    print("Not doing calcuation, requested point to point at cell center (0,0). This is an exact calculation.")
    sys.exit()


if r_gp == 1:
    if centered_gp :
        points = [-h, 0, h]
    else:
        if xstarLoc == 0.5:
            points = [0, h]
        elif xstarLoc == -0.5:
            points = [-h, 0]
elif r_gp == 2:
    if centered_gp:
        points = [-2*h, -h, 0, h, 2*h]
    else:
        if xstarLoc == 0.5:
            points = [-h, 0, h, 2*h]
        elif xstarLoc == -0.5:
            points = [-2*h, -h, 0, h]
elif r_gp == 3:
    if centered_gp:
        points = [-3*h, -2*h, -h, 0, h, 2*h, 3*h]
    else:
        if xstarLoc == 0.5:
            points = [-2*h, -h, 0, h, 2*h, 3*h]
        elif xstarLoc == -0.5:
            points = [-3*h, -2*h, -h, 0, h, 2*h]
            
# Compute kernel matrix numerically
n_points = len(points)
K = np.zeros((n_points, n_points))

for i in range(n_points):
    for j in range(n_points):
        K[i,j] = kernel(points[i], points[j], h)
        
# Compute k_star vector
k_star = np.zeros(n_points)
for i in range(n_points):
    k_star[i] = k_star_kernel(points[i], x_star, h)


# multiply numerically by h^k, so we can symbolically divide by h^k later, k=0,1,2
if der_ordr == 0:
    weights = np.matmul(k_star, np.linalg.inv(K))
elif der_ordr == 1:
    weights = np.matmul(k_star, np.linalg.inv(K))*h
elif der_ordr == 2:
    weights = np.matmul(k_star, np.linalg.inv(K))*h**2

print("weights: ", weights)

# Create symbol
nterms=12


f_3mh = taylor_expansion(nterms,-3*dx, xstarLoc*dx)
f_2mh = taylor_expansion(nterms,-2*dx, xstarLoc*dx)
f_1mh = taylor_expansion(nterms,-1*dx, xstarLoc*dx)
f_0   = taylor_expansion(nterms, 0*dx, xstarLoc*dx)
f_1ph = taylor_expansion(nterms, 1*dx, xstarLoc*dx)
f_2ph = taylor_expansion(nterms, 2*dx, xstarLoc*dx)
f_3ph = taylor_expansion(nterms, 3*dx, xstarLoc*dx)


if r_gp == 1:
    if centered_gp:
        fs = [f_1mh, f_0, f_1ph]
    else:
        if xstarLoc == 0.5:
            fs = [f_0, f_1ph]
        elif xstarLoc == -0.5:
            fs = [f_1mp, f_0]
elif r_gp == 2:
    if centered_gp:
        fs = [f_2mh, f_1mh, f_0, f_1ph, f_2ph]
    else:
        if xstarLoc == 0.5:
            fs = [f_1mh, f_0, f_1ph, f_2ph]
        elif xstarLoc == -0.5:
            fs = [f_2mh, f_1mh, f_0, f_1ph]
elif r_gp == 3:
    if centered_gp:
        fs = [f_3mh, f_2mh, f_1mh, f_0, f_1ph, f_2ph, f_3ph]
    else:
        if xstarLoc == 0.5:
            fs = [f_2mh, f_1mh, f_0, f_1ph, f_2ph, f_3ph]
        elif xstarLoc == -0.5:
            fs = [f_3mh, f_2mh, f_1mh, f_0, f_1ph, f_2ph]
            
pred = np.dot(weights, fs)  #GP prediction value. (taylor expanded)



# instruction to interpreting the output :
# (a) ignore the first few terms with 1/h, 1/h^2, etc.
# (b) the term without h is what GP calculates; its coeff should be very close to 1.
# (c) the GP order = k, which appears in the first term with h^k, k>=1.

if der_ordr == 0:
    pred = simplify(pred)
elif der_ordr == 1:
    pred = simplify(pred/dx) #symbolically divide by h.
elif der_ordr == 2:
    pred = simplify(pred/dx**2) #symbolically divide by h**2.


pred_coll = collect(pred,dx)
print('== GP prediction in Taylor expansion ==')
print(pred_coll)
print()
print()
c0=pred.coeff(dx,0); f0=symbols('f0')
c1=pred.coeff(dx,1); f1=symbols('f1')
c2=pred.coeff(dx,2); f2=symbols('f2')
c3=pred.coeff(dx,3); f3=symbols('f3')
c4=pred.coeff(dx,4); f4=symbols('f4')
c5=pred.coeff(dx,5); f5=symbols('f5')
c6=pred.coeff(dx,6); f6=symbols('f6')
c7=pred.coeff(dx,7); f7=symbols('f7')
c8=pred.coeff(dx,8); f8=symbols('f8')
c9=pred.coeff(dx,9); f9=symbols('f9')
c10=pred.coeff(dx,10); f10=symbols('f10')
c11=pred.coeff(dx,11); f11=symbols('f11')
c12=pred.coeff(dx,12); f12=symbols('f12')


coeff_c=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
funct_f=[f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]
coefficients=np.zeros(nterms)


for i in range(len(coeff_c)-1):
    #print('i=',i)
    if der_ordr == 0:
        coefficients[i] = coeff_c[i]/funct_f[i]
    elif der_ordr == 1:
        coefficients[i] = coeff_c[i]/funct_f[i+1]
    elif der_ordr == 2:
        coefficients[i] = coeff_c[i]/funct_f[i+2]

# formatting in scientific notation
cc0=np.format_float_scientific(coefficients[0], unique=False, precision=8)
cc1=np.format_float_scientific(coefficients[1], unique=False, precision=8)
cc2=np.format_float_scientific(coefficients[2], unique=False, precision=8)
cc3=np.format_float_scientific(coefficients[3], unique=False, precision=8)
cc4=np.format_float_scientific(coefficients[4], unique=False, precision=8)
cc5=np.format_float_scientific(coefficients[5], unique=False, precision=8)
cc6=np.format_float_scientific(coefficients[6], unique=False, precision=8)
cc7=np.format_float_scientific(coefficients[7], unique=False, precision=8)
cc8=np.format_float_scientific(coefficients[8], unique=False, precision=8)
cc9=np.format_float_scientific(coefficients[9], unique=False, precision=8)


print('======== SUMMARY ========')
if centered_gp:
    print('[0] Centered GP stencil with odd (2*r_gp+1) data')
else:
    print('[0] Biased GP stencil with even (2*r_gp) data')

print('[1] r_gp =     ', r_gp)
print('[2] der_ordr = ', der_ordr)
print('[3] x_star =   ', xstarLoc)
print('[4] coefficients:')
print('    coeff for h^0 =', f"{float(cc0):+.8e}",'*',c0/coefficients[0])
print('    coeff for h^1 =', f"{float(cc1):+.8e}",'*',c1/coefficients[1])
print('    coeff for h^2 =', f"{float(cc2):+.8e}",'*',c2/coefficients[2])
print('    coeff for h^3 =', f"{float(cc3):+.8e}",'*',c3/coefficients[3])
print('    coeff for h^4 =', f"{float(cc4):+.8e}",'*',c4/coefficients[4])
print('    coeff for h^5 =', f"{float(cc5):+.8e}",'*',c5/coefficients[5])
print('    coeff for h^6 =', f"{float(cc6):+.8e}",'*',c6/coefficients[6])
print('    coeff for h^7 =', f"{float(cc7):+.8e}",'*',c7/coefficients[7])
print('    coeff for h^8 =', f"{float(cc8):+.8e}",'*',c8/coefficients[8])
print('    coeff for h^9 =', f"{float(cc9):+.8e}",'*',c9/coefficients[9])


# Shift index to search for max:
# (a) for even gp stencils, perfect cancellations happend and it should
#     look up the max coeff starting from the very first coeff, c0
# (b) for odd gp stencils, we shift the index lookup by one to ignore
#     the first coeff, c0   
#if centered_gp:
shiftInd=1
#else:
#    shiftInd=0
    
LC=np.max(np.abs(coefficients[shiftInd:]))
LC=np.format_float_scientific(LC, unique=False, precision=8)
print('[5] The largest coefficient in magnitude = ', LC)


gp_order = np.argmax(np.abs(coefficients[shiftInd:]))+1
print('[6] Expected GP order = ', gp_order)

#print(f"{x:.2e}")
#print(np.format_float_scientific(coefficients[1]))
#print(np.format_float_scientific(coefficients[2]),'*',c2/coefficients[2])
#"{0:+.03f}".format(1.23456)
# Compute effective order of convergence (p_eff)
numerator = sum(k * abs(coefficients[k]) for k in range(1, len(coefficients)))
denominator = sum(abs(coefficients[k]) for k in range(1, len(coefficients)))
p_eff = numerator / denominator

# Print the effective order of convergence
print('[7] Effective GP order of convergence (p_eff) =', f"{p_eff:.8f}")
