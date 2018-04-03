"""
This work is based off of Bridson's "Fluid Simulation", SIGGRAPH 2007 Notes.
https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf
"""

import numpy as np
import scipy as sp

from numba import cuda, jit
from PIL import Image


class FluidQuantity:
    """Stores quantities like density field, velocity field, etc on a staggered MAC grid
    """

    def __init__(self, width, height, offset_x, offset_y, dx):
        self._w = width
        self._h = height
        self._ox = offset_x
        self._oy = offset_y
        self._dx = dx
        self._val = np.zeros(width*height, dtype=np.float64)
        self._buf = np.zeros(width*height, dtype=np.float64)

    def at(self, x, y):
        """Returns value of self at coordinates (x, y) integeres
        """
        return self._val[x + y*self._w]

    def swap(self):
        """Performance hack for calculation advect()
        """
        self._val, self._buf = self._buf, self._val


class FluidSolver:
    def __init__(self, width, height, fluid_density, fluid_quantities):
        """
        fluid_quantities is a list of quantities to advect
        """
        _dx = 1 / min(width, height)
        # Problem Constants
        self._w = width
        self._h = height
        self._dx = _dx
        self._rho = fluid_density

        # Fluid quantities
        # _u, _v are the x, y velocities on the border of a grid
        self._quantites = fluid_quantities
        self._u = FluidQuantity(width+1, height, 0.0, 0.5, _dx)
        self._v = FluidQuantity(width, height+1, 0.5, 0.0, _dx)

        # Buffers for solving fluid equation
        self._neg_div = np.zeros(width*height, dtype=np.float64)
        self._pressure = np.zeros(width*height, dtype=np.float64)

    def step(self, timestep):
        """
        Perform one simulation step with step size = timestep
        """
        # Calc and store negative divergence in buffer
        self.calc_neg_div()
        self.project(timestep)
        self.make_incompressible(timestep)
        self.advect(timestep)

    def calc_neg_div(self):
        """Eq. (4.11) and (4.13)
        This construction is trivially parallizable, no reason to do it in a loop
        """
        scale = 1 / self._dx

        i = 0
        for iy in range(self._h):
            for ix in range(self._w):
                self._neg_div[i] = -scale * (self._u.at(ix+1, iy) - self._u.at(ix, iy) +
                                             self._v.at(ix, iy+1) - self._v.at(ix, iy))
                i += 1

    def project(self, timestep, tol=1e-5, max_iter=600):
        """Eq. (4.19) Calculate pressure from divergence

        Typically this can be done quickly using GEPP, 
        since it's just a Poisson matrix which is thin band.  
        Because it is a Poisson matrix, conjugate gradient is also good.

        This methods implements a Gauss-Seidel  solver, mostly for ease of implementation
        """
        scale = timestep / (self._rho * self._dx * self._dx)

        max_error = 0
        for iter in range(max_iter):
            max_error = 0
            for iy in range(self._h):
                for ix in range(self._w):
                    idx = ix + iy*self._w
                    diag, off_diag = 0, 0

                    # Boundary checking
                    if ix > 0:
                        diag += scale
                        off_diag -= scale * self._pressure[idx-1]
                    if iy > 0:
                        diag += scale
                        off_diag -= scale * self._pressure[idx-self._w]
                    if ix + 1 < self._w:
                        diag += scale
                        off_diag -= scale * self._pressure[idx+1]
                    if iy + 1 < self._h:
                        diag += scale
                        off_diag -= scale * self._pressure[idx-self._w]

                    new_pressure = (self._neg_div[idx] - off_diag) / diag
                    max_error = max(max_error, abs(
                        self._pressure - new_pressure))

            if max_error < tol:
                print('Pressure solver converged in {} iterations, max. error is {}\n'.format(
                    iter, max_error))
                return

        print('Exceeded max. iter. of {}, max. error is {}\n'.format(
            max_iter, max_error))


def main():
    """Runs simulation
    """
    SIZE_X, SIZE_Y = 128, 128
    FLUID_DENSITY = 0.1
    # Section 3.2 discusses how to set this in some detail. Related by CFL condition
    # Here is empirically, as long as it's small enough, it is ok.
    TIMESTEP = 0.005

    frame = np.zeros((SIZE_X, SIZE_Y), dtype=np.int8)


if __name__ == '__main__':
    main()
