import sympy as sym
import numpy as np

def volume_average(g, f_int):
    '''Volume Average Fucntion works on a grid object,
    assumes f_int is the total integral sympy function.'''

    if g.ndim == 1:
        x = sym.symbols('x')
        f_int = sym.lambdify(x, f_int, 'numpy')
        return (f_int(g.x+0.5*g.dx) - f_int(g.x-0.5*g.dx))/g.dx

    elif g.ndim == 2:
        x,y = sym.symbols('x y')
        f_int = sym.lambdify((x,y), f_int, 'numpy')
        return   (f_int(g.gridX+0.5*g.dx, g.gridY+0.5*g.dy) #F_bd
                - f_int(g.gridX+0.5*g.dx, g.gridY-0.5*g.dy) #F_bc
                - f_int(g.gridX-0.5*g.dx, g.gridY+0.5*g.dy) #F_ad
                + f_int(g.gridX-0.5*g.dx, g.gridY-0.5*g.dy))/(g.dx*g.dy)

def K_se(x,y,ell):
    x = np.array(x)
    y = np.array(y)

    return np.exp( - np.linalg.norm(x-y,ord=2) / ell**2)

def get_gp_stencil(grid, index, r_gp, method='diamond'):

    if method == 'diamond':
        array = []

        ncols = grid.shape[1]
        nrows = grid.shape[0]

        center_col = index[1]
        center_row = index[0]


        top = (center_row-r_gp, center_col)
        #width on top starts at 1
        width = 1

        for i in range(index[0]-r_gp, index[0]+r_gp+1):

            each_side = int((width+1)/2 - 1)
            for j in range(index[1]-each_side, index[1]+each_side+1):

                array.append(grid[i,j])
            if i-index[0]+r_gp < r_gp:
                width += 2
            else:
                if i-index[0]+r_gp == 2*r_gp + 1:
                    break
                else:
                    width -= 2
    return np.array(array)

    
