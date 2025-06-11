import numpy as np
import matplotlib.pyplot as plt

class Grid1D:
    def __init__(self, xlim, Nx, Nghost):
        self.xlim = xlim
        self.Nx = Nx
        self.Nghost = Nghost
        self.dx = (xlim[1]-xlim[0])/(Nx-1)
        self.x = np.linspace(xlim[0] - self.dx*Nghost, xlim[1] + self.dx*Nghost, Nx + 2*Nghost)
        self.ndim = 1
        #internal x (no ghost cells)
        self.x_int = self.x[self.Nghost:self.Nx+self.Nghost]


    def fill_grid(self, f):
        self.grid = f(self.x)
        self.internal_grid = self.grid[self.Nghost:self.Nx+self.Nghost]

    def fill_grid_data(self, y):
        assert(len(y) == self.Nx)
        self.grid = np.zeros_like(self.x)
        self.grid[self.Nghost:self.Nx+self.Nghost] = y
        self.internal_grid = self.grid[self.Nghost:self.Nx+self.Nghost]

    def plot(self):
        fig = plt.figure()
        plt.plot(self.x, self.grid)
        plt.show()


    def apply_periodic_bcs(self):

        for i in range(self.Nghost):
            self.grid[i] = self.grid[self.Nx + self.Nghost - (self.Nghost-i)]
            self.grid[self.Nx + self.Nghost + i] = self.grid[self.Nghost + i]

    def return_internal_grid(self):
        return self.grid[self.Nghost:self.Nx+self.Nghost]





class Grid2D:
    def __init__(self, xlim, ylim, Nx, Ny, Nghost):
        self.xlim = xlim
        self.ylim = ylim

        self.Nx = Nx
        self.Ny = Ny
        self.Nghost = Nghost
        self.ndim = 2

        self.dx = (xlim[1]-xlim[0])/(Nx-1)
        self.dy = (ylim[1]-ylim[0])/(Ny-1)

        self.x = np.linspace(xlim[0] - self.dx*Nghost, xlim[1] + self.dx*Nghost, Nx + 2*Nghost)
        self.y = np.linspace(ylim[0] - self.dy*Nghost, ylim[1] + self.dy*Nghost, Ny + 2*Nghost)

        self.x_int = self.x[self.Nghost:self.Nx+self.Nghost]
        self.y_int = self.y[self.Nghost:self.Ny+self.Nghost]

        self.gridX, self.gridY = np.meshgrid(self.x, self.y)



    def fill_grid(self, f):
        self.grid = f(self.gridX, self.gridY)
        self.internal_grid = self.grid[self.Nghost:self.Nx+self.Nghost, self.Nghost:self.Ny+self.Nghost]
    
    def fill_grid_data(self, y):
        assert(len(y) == self.Nx)
        self.grid = np.zeros_like(self.x)
        self.grid[self.Nghost:self.Nx+self.Nghost] = y
        self.internal_grid = self.grid[self.Nghost:self.Nx+self.Nghost]

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        axp = ax.imshow(self.internal_grid, cmap = 'ocean')
        cb = plt.colorbar(axp,ax=[ax],location='right')
        plt.show()



class Grid3D:
    def __init__(self, xlim, ylim, zlim, Nx, Ny, Nz, Nghost):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.Nx = Nx
        self.Ny = Ny 
        self.Nz = Nz

        self.Nghost = Nghost
        self.ndim = 3

        self.dx = (xlim[1]-xlim[0])/(Nx-1)
        self.dy = (ylim[1]-ylim[0])/(Ny-1)
        self.dz = (zlim[1]-zlim[0])/(Nz-1)

        self.x = np.linspace(xlim[0] - self.dx*Nghost, xlim[1] + self.dx*Nghost, Nx + 2*Nghost)
        self.y = np.linspace(ylim[0] - self.dy*Nghost, ylim[1] + self.dy*Nghost, Ny + 2*Nghost)
        self.z = np.linspace(zlim[0] - self.dz*Nghost, zlim[1] + self.dz*Nghost, Nz + 2*Nghost)

        self.x_int = self.x[self.Nghost:self.Nx+self.Nghost]
        self.y_int = self.y[self.Nghost:self.Ny+self.Nghost]
        self.z_int = self.z[self.Nghost:self.Nz+self.Nghost]

        self.gridX, self.gridY, self.gridZ= np.meshgrid(self.x, self.y, self.z)



    def fill_grid(self, f):
        self.grid = f(self.gridX, self.gridY, self.gridZ)
        self.internal_grid = self.grid[self.Nghost:self.Nx+self.Nghost, self.Nghost:self.Ny+self.Nghost, self.Nghost:self.Nz+self.Nghost]

