import multiprocessing as mp
import os
import time


from sin_derivatives import sin_derivatives
from ci_plot import ci_plot
from kernel_comparison_shock import kernel_comparison_shock
from compound_waves import compound_waves
from shock_2D import shock_2d
from showcase_2D import showcase2D
from convergence_tables1D import convergence_tables1D
from convergence_tables2D import convergence_tables2D
from condition_number import condition_number
from heatmap import kernel_heat_map

if __name__ == "__main__":
    start_time = time.time()

    sin_derivatives()
    ci_plot()
    kernel_comparison_shock()
    condition_number()
    showcase2D()
    compound_waves() 
    kernel_heat_map()
    convergence_tables2D()
    convergence_tables1D()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Execution Time:", elapsed_time, "seconds")






