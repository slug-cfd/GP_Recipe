import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import matplotlib as mpl
from cmcrameri import cm  # Scientific colormaps

def kernel_heat_map():
    # Set global aesthetics for publication quality
    # plt.rcParams.update({
    #     #'font.family': 'serif',
    #     #'font.serif': ['Computer Modern Roman'],
    #     #'text.usetex': False,
    #     'axes.labelsize': 16,
    #     'axes.titlesize': 18,
    #     'xtick.labelsize': 14,
    #     'ytick.labelsize': 14,
    #     'figure.figsize': (12, 10),
    #     'figure.dpi': 300,
    #     'savefig.dpi': 400,
    #     'savefig.bbox': 'tight',
    #     'savefig.pad_inches': 0.05
    # })

    # Kernel functions
    def squared_exponential(x, y, length_scale=1.0):
        """Squared Exponential kernel function"""
        return np.exp(-0.5 * (x - y)**2 / length_scale**2)
    
    def neural_network(x, y):
        """Neural Network kernel function"""
        sigma0 = 1
        sigma = 1
        inner_prod = 2 * (sigma0**2 + sigma**2 * x * y) / (
            np.sqrt(1 + 2*(sigma0**2 + sigma**2 * x**2)) * 
            np.sqrt(1 + 2*(sigma0**2 + sigma**2 * y**2))
        )
        return 2 / np.pi * np.arcsin(inner_prod)

    def discontinuous_arcsin(x, y):
        """Discontinuous ArcSin kernel function"""
        sigma0 = 1
        sigma = 1
        d = np.sqrt((x-y)**2)
        inner_prod = np.exp(-d) / (1 + 2*(sigma0**2 + sigma**2 * (x-y)**2))
        return 2 / np.pi * np.arcsin(inner_prod)

    # Generate grid points
    resolution = 80  # Higher resolution for smoother visualization
    x = np.linspace(-5, 5, resolution)
    y = np.linspace(-5, 5, resolution)
    X, Y = np.meshgrid(x, y)

    # Compute kernel matrices
    kernels = {
        r"Squared Exponential ($\ell=1.0$)": squared_exponential(X, Y),
        r"Squared Exponential ($\ell=0.35$)": squared_exponential(X, Y, length_scale=0.35),
        r"Neural Network ($\sigma_0 = \sigma = 1$)": neural_network(X, Y),
        r"Discontinuous ArcSin": discontinuous_arcsin(X, Y)
    }

    # Create figure with GridSpec for better control - ultra compact figure with colorbar on right
    fig = plt.figure(figsize=(11, 9))
    # Create a layout with 2 rows, 3 columns where the 3rd column is very narrow for the colorbar
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.04])
    gs.update(wspace=0.015, hspace=0.12)  # Slightly more spacing

    # Select a scientific colormap for better visualization
    #cmap = cm.lajolla  # La Jolla colormap, or try: batlow, oslo, berlin
    cmap = "turbo"

    # Plot each kernel
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (row, col) for each subplot
    axes = []
    
    for i, ((title, kernel_matrix), pos) in enumerate(zip(kernels.items(), subplot_positions)):
        ax = plt.subplot(gs[pos])
        axes.append(ax)
        
        # Plot heatmap with improved aesthetics
        im = ax.imshow(
            kernel_matrix, 
            cmap=cmap,
            extent=[-5, 5, -5, 5],
            vmin=0, 
            vmax=1,
            origin='lower',
            interpolation='bicubic',  # Smoother interpolation
            aspect='equal'
        )
        
        # Set titles and labels with reduced padding
        ax.set_title(title, fontsize=16, pad=8)
        
        # Minimal, elegant ticks
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        if i % 2 == 0:  # Only show y-label on left plots
            ax.set_ylabel(r"$\mathbf{y}$", fontsize=14, labelpad=3)
        else:
            ax.set_yticklabels([])
            
        if i >= 2:  # Only show x-label on bottom plots
            ax.set_xlabel(r"$\mathbf{x}$", fontsize=14, labelpad=3)
        else:
            ax.set_xticklabels([])
            
        # Add subtle grid
        ax.grid(alpha=0.15, linestyle='--', linewidth=0.5)
        
        # Add a thin border
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('gray')

    # Create colorbar on the right side that spans both rows
    cbar_ax = plt.subplot(gs[:, 2])  # Spans all rows in the last column
    
    cbar = fig.colorbar(
        im, 
        cax=cbar_ax, 
        orientation='vertical',
        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    cbar.ax.tick_params(labelsize=12)
    
    # Position the colorbar label to avoid overlap
    cbar.ax.set_title('$k(x,y)$', fontsize=14, pad=8)
    
    # Save figure with high resolution and minimal padding
    plt.savefig("heat_map_kernels.png", dpi=400, bbox_inches='tight', pad_inches=0.03)
    plt.show()

if __name__ == "__main__":
    kernel_heat_map()


# Z = SE(X, Y)

# plt.figure(figsize=(8, 6))
# sns.heatmap(Z, cmap='viridis')
# plt.title('Squared Exponential Kernel Heat Map')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.xticks([])
# plt.yticks([])
# plt.savefig("heat_map_SE.png", dpi=800)


# Z = NN(X, Y)

# plt.figure(figsize=(8, 6))
# sns.heatmap(Z, cmap='viridis')
# plt.title('Neural Network Kernel Heat Map')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.xticks([])
# plt.yticks([])
# plt.savefig("heat_map_NN.png", dpi=800)


# Z = DAS(X, Y)

# plt.figure(figsize=(8, 6))
# sns.heatmap(Z, cmap='viridis')
# plt.title('Discontinous Arcsin Kernel Heat Map')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.xticks([])
# plt.yticks([])
# plt.savefig("heat_map_DAS.png", dpi=800)
