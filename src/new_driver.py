import sympy as sym
import numpy as np
import inspect
import sys
from grids import Grid1D, Grid2D
from tools import volume_average, get_gp_stencil
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mpmath as mm
import mymath as math
import kernels as kern
from mymath import MathModule

class GP_recipe1D:

    def __init__(self, grid, r_gp, **kwargs):

        #ell=0.8, nu=1.5, stencil_method="center", high_precision=False):
        self.g = grid
        self.r_gp = r_gp
        self.ell = kwargs.get("ell", 0.8)
        self.nu = kwargs.get("nu", 1.5)
        self.stencil_method = kwargs.get("stencil_method", "center")
        self.high_precision = kwargs.get("high_precision", False)
        self.mean_method = kwargs.get("mean_method", "zero")
        self.precision = kwargs.get("precision", 50)
        self.dx = self.g.x[1] - self.g.x[0]
        self.kwargs = kwargs
        self.KernelFunction = None
        self.KStarFunction = None
        if self.high_precision:
            mm.mp.prec = self.precision

        assert(grid.ndim == 1)

        self.math = MathModule(self.high_precision)


    #define stencil method...
    def get_stencil(self, x_i):
        internal_x = self.g.x[self.r_gp:-self.r_gp]
        closest_index = np.argmin(np.abs(internal_x - x_i)) + self.r_gp

        if self.stencil_method == "center":
            stencilx = self.g.x[closest_index-self.r_gp:closest_index+self.r_gp+1]
            stencily = self.g.grid[closest_index-self.r_gp:closest_index+self.r_gp+1]

        else:
            if self.g.x[closest_index] >= x_i:  # grid point is to the right
                if self.stencil_method == "left":
                    end = closest_index
                    start = end - self.r_gp
                elif self.stencil_method == "right":
                    start = closest_index
                    end = start + self.r_gp

            else:  # grid point is to the left
                if self.stencil_method == "left":
                    end = closest_index + 1
                    start = end - self.r_gp
                elif self.stencil_method == "right":
                    start = closest_index + 1
                    end = start + self.r_gp

            stencilx = self.g.x[start:end]
            stencily = self.g.grid[start:end]

        if self.high_precision:
            stencilx = mm.matrix(stencilx.tolist())
            stencily = mm.matrix(stencily.tolist())

        return stencilx, stencily

    

    def get_K(self, f, x, dx):
        N = len(x)
        K = self.math.zeros_matrix(N,N)
        for i in range(N):
            for j in range(N):

                if self.high_precision and x.cols > 1:
                    K[i,j] = f(x[i, :], x[j, :], dx, **self.kwargs)
                else:                    
                    K[i,j] = f(x[i], x[j], dx, **self.kwargs)
        return K

    def get_KStar(self, f, x, xstar, dx):
        N = len(x)
        k_star = self.math.zeros_matrix(1,N)

        for i in range(N):
            if self.high_precision and x.cols > 1:
                k_star[i] = f(x[i, :], xstar, dx, **self.kwargs)
            else:        
                k_star[i] = f(x[i], xstar, dx, **self.kwargs)

        return k_star

    #Also returns the covariance for calculating confidence intervals
    def convert_custom_ci(self, x_i, Kf, K_starf, K_starstarf):

        if np.isscalar(x_i):
            x_i = [x_i]

        sol = np.zeros_like(x_i)
        sigmas = np.zeros(len(x_i))
        if self.high_precision:
            sigmas = mm.matrix(sigmas)
        
        for i, x_star in enumerate(x_i):

            stencilx, stencily = self.get_stencil(x_star)

            if self.high_precision:
                stencilx = mm.matrix(stencilx)
                x_star = mm.mpf(x_star)
                self.dx = mm.mpf(self.dx)
                self.ell = mm.mpf(self.ell)

        
            # KernelFunction = kern.get_sympy_kernel_1D(Kf)
            # KStarFunction = kern.get_sympy_kernel_1D(K_starf)
            # KStarStarFunction = kern.get_sympy_kernel_1D(K_starstarf)

            KernelFunction = kern.SympyKernel1D(Kf, **self.kwargs)        
            KStarFunction = kern.SympyKernel1D(K_starf, **self.kwargs)
            KStarStarFunction = kern.SympyKernel1D(K_starstarf, **self.kwargs)

            K = self.get_K(KernelFunction, stencilx, self.dx)
            K_star = self.get_KStar(KStarFunction, stencilx, x_star, self.dx)

            try:
                invK = self.math.inv(K)
            except:
                # if self.high_precision:
                #     #mpmath doesn't have pinv.
                #     current_prec = mm.mp.dps
                #     mm.mp.dps = 500
                #     print("current precision: ", current_prec)
                #     print(np.array(K.tolist))
                #     invK = self.math.inv(K)
                #     mm.mp.dps = current_prec
                # else:
                #     #implement pinv if needed for not highprecision.
                print("Not enough precision to invert...")
                exit()        

            #K_* K^-1
            gp_pred = self.math.matmul(K_star, invK)

            if self.mean_method == "optimal":
                #gp_mean was calculated and passed in. Use it
                N = K.shape[0]
                top = self.math.matmul( self.math.matmul(self.math.ones(N).T, invK), stencily)
                bottom = self.math.matmul(self.math.matmul(self.math.ones(N).T,  invK), self.math.ones(N))
                gp_mean = top/bottom
            elif self.mean_method == "simple":
                gp_mean = self.math.mean(stencily)
            elif self.mean_method == "zero":
                gp_mean = 0.0
            else:
                print("mean_method not supported.")
                exit()
            
        
            stencily = stencily - self.math.ones(len(stencily))*gp_mean

            sol_i = gp_mean + self.math.matmul(gp_pred, stencily)

            if self.high_precision:
                #this will result in a mpmath.matrix object of 1 number, so just grap the number.
                sol_i = sol_i[0]
            

            #compute covariance.
            #Σ* = K** - K*^T * K^-1 * K*
            K_starstar = KStarStarFunction(x_star, x_star, self.dx)

            sol[i] = float(sol_i)

            #i hate mpmath
            if self.high_precision:
                sigma = K_starstar - self.math.matmul(self.math.matmul(K_star,self.math.inv(K)), K_star.T)
                sigma = sigma[0]
                sigma = sigma.real
            else:
                sigma = K_starstar - self.math.matmul(self.math.matmul(K_star.T,self.math.inv(K)), K_star)
    
            sigmas[i] = sigma


        return sol, sigmas


    def convert_custom(self, x_i, Kf, K_starf, stationary=False, hack_for_getting_K=False):

        if np.isscalar(x_i):
            x_i = [x_i]

        # Initialize sol as a 1D array of the correct length
        sol = np.zeros(len(x_i))

        for i, x_star in enumerate(x_i):

            stencilx, stencily = self.get_stencil(x_star)

            if self.high_precision:
                stencilx = mm.matrix(stencilx)
                x_star = mm.mpf(x_star)
                self.dx = mm.mpf(self.dx)
                self.ell = mm.mpf(self.ell)

        
            if (not self.KernelFunction) or (not stationary):
                #self.KernelFunction = kern.get_sympy_kernel_1D(Kf)
                self.KernelFunction = kern.SympyKernel1D(Kf, **self.kwargs)
            
            if (not self.KStarFunction) or (not stationary):
                #self.KStarFunction = kern.get_sympy_kernel_1D(K_starf)
                self.KStarFunction = kern.SympyKernel1D(K_starf, **self.kwargs)
            
            K = self.get_K(self.KernelFunction, stencilx, self.dx)

            K_star = self.get_KStar(self.KStarFunction, stencilx, x_star, self.dx)

            #just wanted to hack this to get K out so we can do a condition 
            #number plot.
            if hack_for_getting_K:
                return K, K_star

            

            invK = self.math.inv(K)


            #K_* K^-1
            gp_pred = self.math.matmul(K_star, invK)

            previouslyWasHighPrecision = self.high_precision

            if self.high_precision:
                #convert back to low precision now.
                 gp_pred = np.array(gp_pred.tolist())
                 gp_pred = np.array([[float(element) for element in row] for row in gp_pred], dtype=np.float64) 
                 self.high_precision = False
                 self.math.high_precision = False


            if self.mean_method == "optimal":
                #gp_mean was calculated and passed in. Use it
                N = K.shape[0]
                top = self.math.matmul( self.math.matmul(self.math.ones(N).T, invK), stencily)
                bottom = self.math.matmul(self.math.matmul(self.math.ones(N).T,  invK), self.math.ones(N))
                gp_mean = top/bottom
            elif self.mean_method == "simple":
                gp_mean = self.math.mean(stencily)
            elif self.mean_method == "zero":
                gp_mean = 0.0
            else:
                print("mean_method not supported.")
                exit()
            
            
            #stencily = stencily.T
            

            if (not self.high_precision) and (previouslyWasHighPrecision):
                #we need to lower order of stencily.
                stencily = np.array(stencily.tolist())
                stencily = np.array([float(element) for element in stencily], dtype=np.float64)


            stencily = stencily - self.math.ones(len(stencily))*gp_mean

            sol_i = gp_mean + self.math.matmul(gp_pred, stencily)

            if self.high_precision:
                #this will result in a mpmath.matrix object of 1 number, so just grap the number.
                sol_i = sol_i[0]
            
            
            sol[i] = float(sol_i)


            #not proud of this one, but it'll do
            if previouslyWasHighPrecision:
                self.high_precision = True
                self.math.high_precision = True

        return sol




class GP_recipe2D:

    def __init__(self, grid, r_gp, **kwargs):
    
        self.g = grid
        self.r_gp = r_gp
        self.ell = kwargs.get("ell", 0.8)
        self.stencil_method = kwargs.get("stencil_method", "diamond")
        self.high_precision = kwargs.get("high_precision", False)
        self.mean_method = kwargs.get("mean_method", "zero")
        self.precision = kwargs.get("precision", 50)
        self.dx = self.g.x[1] - self.g.x[0]
        self.dy = self.g.y[1] - self.g.y[0]
        self.dxdy = self.dx*self.dy
        self.kwargs = kwargs
        self.KernelFunction = None
        self.KStarFunction = None
        self.gp_pred = None #if stationary, we can just store this and reuse it.
        self.invK = None
        if self.high_precision:
            mm.mp.prec = self.precision


        assert(grid.ndim == 2)
        self.math = MathModule(self.high_precision)

    def get_stencil(self,x_i):
        #x_i is the point we want to interpolate at.
        assert(len(x_i) == self.g.ndim)

        internal_x = self.g.x[self.r_gp:-self.r_gp]
        
        assert(x_i[0] >= internal_x[0])
        assert(x_i[0] <= internal_x[-1])

        internal_y = self.g.y[self.r_gp:-self.r_gp]
        assert(x_i[1] >= internal_y[0])
        assert(x_i[1] <= internal_y[-1])

        closest_index = [0,0]
        closest_index[0] = np.argmin(np.abs(internal_x-x_i[0])) + self.r_gp
        closest_index[1] = np.argmin(np.abs(internal_y-x_i[1])) + self.r_gp

        if self.stencil_method == 'diamond':
            stencilgrid = []
            stencilx = []
            stencily = []

            ncols = self.g.grid.shape[1]
            nrows = self.g.grid.shape[0]

            center_col = closest_index[1]
            center_row = closest_index[0]



            top = (center_row-self.r_gp, center_col)
            #width on top starts at 1
            width = 1

            for i in range(closest_index[0]-self.r_gp, closest_index[0]+self.r_gp+1):

                each_side = int((width+1)/2 - 1)
                for j in range(closest_index[1]-each_side, closest_index[1]+each_side+1):

                    stencilgrid.append(self.g.grid[j,i])
                    stencilx.append(self.g.gridX[j,i])
                    stencily.append(self.g.gridY[j,i])
                if i-closest_index[0]+self.r_gp < self.r_gp:
                    width += 2
                else:
                    if i-closest_index[0]+self.r_gp == 2*self.r_gp + 1:
                        break
                    else:
                        width -= 2
        elif self.stencil_method == 'square':
            stencilgrid = []
            stencilx = []
            stencily = []

            ncols = self.g.grid.shape[1]
            nrows = self.g.grid.shape[0]

            center_col = closest_index[1]
            center_row = closest_index[0]


            top = (center_row-self.r_gp, center_col)
            #width on top starts at 1
            width = 1

            for i in range(closest_index[0]-self.r_gp, closest_index[0]+self.r_gp+1):                
                for j in range(closest_index[1]-self.r_gp, closest_index[1]+self.r_gp+1):

                    stencilgrid.append(self.g.grid[j,i])
                    stencilx.append(self.g.gridX[j,i])
                    stencily.append(self.g.gridY[j,i])

        elif self.stencil_method == 'blocky_diamond':
            stencilgrid = []
            stencilx = []
            stencily = []

            ncols = self.g.grid.shape[1]
            nrows = self.g.grid.shape[0]

            center_col = closest_index[1]
            center_row = closest_index[0]


            top = (center_row-self.r_gp, center_col)
            #width on top starts at 1
            width = 1

            for i in range(closest_index[0]-self.r_gp, closest_index[0]+self.r_gp+1):                
                for j in range(closest_index[1]-self.r_gp, closest_index[1]+self.r_gp+1):
                    

                    isCorner =  ((i == closest_index[0] - self.r_gp and j == closest_index[1] - self.r_gp) or 
                                (i == closest_index[0] + self.r_gp and j == closest_index[1] - self.r_gp) or 
                                (i == closest_index[0] - self.r_gp and j == closest_index[1] + self.r_gp) or 
                                (i == closest_index[0] + self.r_gp and j == closest_index[1] + self.r_gp))

                    #block diamond with r_gp =1 is square.
                    if self.r_gp <3 :

                        if self.r_gp == 1 or not isCorner:
                            stencilgrid.append(self.g.grid[j,i])
                            stencilx.append(self.g.gridX[j,i])
                            stencily.append(self.g.gridY[j,i])

                    elif self.r_gp == 3:
                        # Fixed: Removed the redundant inner loops that were here previously.
                        # The 'isCorner' check correctly uses the i, j from the outer loops.
                        if not isCorner:
                            stencilgrid.append(self.g.grid[j, i])
                            stencilx.append(self.g.gridX[j, i])
                            stencily.append(self.g.gridY[j, i])


                    else:
                        print(f"blocky_diamond stencil not implemented for r_gp: {self.r_gp}")
                        
        
                    

        elif self.stencil_method == 'biased_diamond':
            stencilgrid = []
            stencilx = []
            stencily = []

            ncols = self.g.grid.shape[1]
            nrows = self.g.grid.shape[0]

            center_col = closest_index[1]
            center_row = closest_index[0]


            top = (center_row-self.r_gp, center_col)
            #width on top starts at 1
            width = 1

            for i in range(closest_index[0]-self.r_gp, closest_index[0]+self.r_gp+1):

                each_side = int((width+1)/2 - 1)
                for j in range(closest_index[1]-each_side, closest_index[1]+each_side+1):

                    stencilgrid.append(self.g.grid[j,i])
                    stencilx.append(self.g.gridX[j,i])
                    stencily.append(self.g.gridY[j,i])
                if i-closest_index[0]+self.r_gp < self.r_gp:
                    width += 2
                else:
                    if i-closest_index[0]+self.r_gp == 2*self.r_gp + 1:
                        break
                    else:
                        width -= 2
                
            #ok now lets just go back through and add the upper left corner of square.
            for i in range(closest_index[0], closest_index[0]+self.r_gp+1):                
                for j in range(closest_index[1], closest_index[1]+self.r_gp+1):
                    
                    # Check if this point is already included.
                    if (self.g.gridX[j,i], self.g.gridY[j,i]) not in zip(stencilx, stencily):
                        stencilgrid.append(self.g.grid[j,i])
                        stencilx.append(self.g.gridX[j,i])
                        stencily.append(self.g.gridY[j,i])

        elif self.stencil_method == 'cross':
            stencilgrid = []
            stencilx = []
            stencily = []

            ncols = self.g.grid.shape[1]
            nrows = self.g.grid.shape[0]

            center_col = closest_index[1]
            center_row = closest_index[0]

            # Horizontal part of the cross (including center)
            for i in range(closest_index[0]-self.r_gp, closest_index[0]+self.r_gp+1):                
                j = closest_index[1]
                stencilgrid.append(self.g.grid[j,i])
                stencilx.append(self.g.gridX[j,i])
                stencily.append(self.g.gridY[j,i])

            # Vertical part of the cross (excluding center)
            for j in range(closest_index[1]-self.r_gp, closest_index[1]+self.r_gp+1):
                i = closest_index[0]
                if j != closest_index[1]:  # Skip the center point
                    stencilgrid.append(self.g.grid[j,i])
                    stencilx.append(self.g.gridX[j,i])
                    stencily.append(self.g.gridY[j,i])

        elif self.stencil_method == "minimal":

            #copy and paste cross as starting point.
            stencilgrid = []
            stencilx = []
            stencily = []

            ncols = self.g.grid.shape[1]
            nrows = self.g.grid.shape[0]

            center_col = closest_index[1]
            center_row = closest_index[0]

            # Horizontal part of the cross (including center)
            for i in range(closest_index[0]-self.r_gp, closest_index[0]+self.r_gp+1):                
                j = closest_index[1]
                stencilgrid.append(self.g.grid[j,i])
                stencilx.append(self.g.gridX[j,i])
                stencily.append(self.g.gridY[j,i])

            # Vertical part of the cross (excluding center)
            for j in range(closest_index[1]-self.r_gp, closest_index[1]+self.r_gp+1):
                i = closest_index[0]
                if j != closest_index[1]:  # Skip the center point
                    stencilgrid.append(self.g.grid[j,i])
                    stencilx.append(self.g.gridX[j,i])
                    stencily.append(self.g.gridY[j,i])


            i = closest_index[0]
            j = closest_index[1]
            
            #add the extras.
            if self.r_gp == 1:
                stencilgrid.append(self.g.grid[j+1,i+1])
                stencilx.append(self.g.gridX[j+1,i+1])
                stencily.append(self.g.gridY[j+1,i+1])

                stencilgrid.append(self.g.grid[j-1,i+1])
                stencilx.append(self.g.gridX[j-1,i+1])
                stencily.append(self.g.gridY[j-1,i+1])


            elif self.r_gp == 2:
                stencilgrid.append(self.g.grid[j+1,i+1])
                stencilx.append(self.g.gridX[j+1,i+1])
                stencily.append(self.g.gridY[j+1,i+1])

                stencilgrid.append(self.g.grid[j+2,i+1])
                stencilx.append(self.g.gridX[j+2,i+1])
                stencily.append(self.g.gridY[j+2,i+1])


                stencilgrid.append(self.g.grid[j-1,i+1])
                stencilx.append(self.g.gridX[j-1,i+1])
                stencily.append(self.g.gridY[j-1,i+1])

                stencilgrid.append(self.g.grid[j-2,i+1])
                stencilx.append(self.g.gridX[j-2,i+1])
                stencily.append(self.g.gridY[j-2,i+1])


            else:
                print(f"new_idea stencil not implemented for r_gp: {self.r_gp}")

        else:
            print("Stencil not supported")
            exit()
        stencilx = np.array(stencilx)
        stencily = np.array(stencily)
        stencilgrid  = np.array(stencilgrid)
        return stencilx, stencily, stencilgrid



    def get_K(self, f, points, dx, dy):
        N = len(points)
        K = self.math.zeros_matrix(N,N)

  
        if self.high_precision:
            points = mm.matrix(points)
        
        # print(points)
        

        for i in range(N):
            for j in range(N):
                if self.high_precision and len(points) > 1:
                    #TODO
                    K[i,j] = f(points[i, :], points[j, :], dx, dy,**self.kwargs)
                else:                    
                    K[i,j] = f(points[i], points[j], dx, dy, **self.kwargs)
        return K

    def get_KStar(self, f, points, xstar, dx, dy):
        N = len(points)
        k_star = self.math.zeros_matrix(1,N)
  
        if self.high_precision:
            points = mm.matrix(points)

        for i in range(N):
            if self.high_precision and len(points) > 1:
                #TODO
                k_star[i] = f(xstar, points[i, :], dx, dy, **self.kwargs)
            else:        
                k_star[i] = f(xstar, points[i], dx, dy, **self.kwargs)

        return k_star

    def convert_custom(self, x_i, Kf, K_starf, stationary=False, hack_for_getting_K=False):

        if np.isscalar(x_i):
            x_i = [x_i]

        # Initialize sol as a 1D array of the correct length
        sol = np.zeros(len(x_i))

        for i, x_star in enumerate(x_i):

            stencilx, stencily, stencilgrid = self.get_stencil(x_star)
            
            stencil = [[x, y] for x, y in zip(stencilx, stencily)]
            

            if self.high_precision:
                stencilx = mm.matrix(stencilx)
                stencily = mm.matrix(stencily)
                stencilgrid = mm.matrix(stencilgrid)
                x_star = mm.matrix(x_star)
                self.dx = mm.mpf(self.dx)
                self.ell = mm.mpf(self.ell)


            
            if (self.invK is None) or (not stationary): 
                self.KernelFunction = kern.SympyKernel2D(Kf, **self.kwargs)
                K = self.get_K(self.KernelFunction, stencil, self.dx, self.dy)
                cond_number = self.math.cond(K)


                if not hack_for_getting_K: #don't compute inverse if we just want K
                    self.invK = self.math.inv(K)
                #K_* K^-1

            if not hack_for_getting_K: 
                invK = self.invK


            self.KStarFunction = kern.SympyKernel2D(K_starf, **self.kwargs)


            K_star = self.get_KStar(self.KStarFunction, stencil, x_star, self.dx, self.dy)

            if hack_for_getting_K:
                return K, K_star

            self.gp_pred = self.math.matmul(K_star, invK)





            if self.mean_method == "optimal":
                #gp_mean was calculated and passed in. Use it
                N = K.shape[0]
                top = self.math.matmul( self.math.matmul(self.math.ones(N).T, invK), stencilgrid)
                bottom = self.math.matmul(self.math.matmul(self.math.ones(N).T,  invK), self.math.ones(N))
                gp_mean = top/bottom
            elif self.mean_method == "simple":
                gp_mean = self.math.mean(stencilgrid)
            elif self.mean_method == "zero":
                gp_mean = 0.0
            else:
                print("mean_method not supported.")
                exit()

        
            
            stencilgrid = stencilgrid - self.math.ones(len(stencilgrid))*gp_mean

            sol_i = gp_mean + self.math.matmul(self.gp_pred, stencilgrid)

            if self.high_precision:
                #this will result in a mpmath.matrix object of 1 number, so just grap the number.
                sol_i = sol_i[0]
            
            sol[i] = float(sol_i)


        return sol



