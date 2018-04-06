"""
This work is based off of Bridson's "Fluid Simulation", SIGGRAPH 2007 Notes.
https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf
"""

import math
import numpy as np

from PIL import Image


class FluidQuantity:
    """Stores quantities like density field, velocity field, etc
    on a staggered MAC grid
    """

    def __init__(self, size, offset, dx):
        self._w = size[0]
        self._h = size[1]
        self._ox = offset[0]
        self._oy = offset[1]
        self._dx = dx
        self._val = np.zeros(self._w*self._h, dtype=np.float64)
        self._buf = np.zeros(self._w*self._h, dtype=np.float64)

    def at(self, x, y):
        """Returns value of self at coordinates (x, y) integeres
        """
        return self._val[x + y*self._w]

    def apply(self, x, y, f):
        """
        Apply function f to value at (x, y).
        f(x) the first argument is the current value at (x, y).
        """
        self._val[x + y*self._w] = f(self._val[x + y*self._w])

    def swap(self):
        """Performance hack for calculation advect()
        """
        self._val, self._buf = self._buf, self._val

    def advect(self, timestep, u, v):
        """Advect the fluid quantity use velocity fields of u, v

        Eq. (3.6) to (3.9).  This is the Lagrangian part of the
        semi-Lagrangian methods.
        """
        idx = 0
        for iy in range(self._h):
            for ix in range(self._w):
                # Need these in this loop because they'll get changed
                x = ix + self._ox
                y = iy + self._oy

                # Forward Euler method
                # Divide by dx because we want an index value
                x -= u.linear_interpolate(x, y) * timestep / self._dx
                y -= v.linear_interpolate(x, y) * timestep / self._dx

                # We now know the value at the next time step should be
                # the old value at (x, y), however this might not be perfectly
                # on a grid point, so we interpolate again.
                # Writing to buf[] because we still need to use the old values
                # to advect other quantities.
                self._buf[idx] = self.linear_interpolate(x, y)
                idx += 1

    def linear_interpolate(self, x, y):
        """2D linear interpolate the quantity values at (x, y) from four
        near by grid points.
        """
        # Boundary checking
        x = min(max(x - self._ox, 0.0), self._w - 1.001)
        y = min(max(y - self._oy, 0.0), self._h - 1.001)
        # Extract integer and fractional parts
        # Modf returned signed values, but it's ok since we check the
        # bounds above, so x >= 0 and y >= 0
        fx, ix = math.modf(x)
        fy, iy = math.modf(y)

        x00 = self.at(ix+0, iy+0)
        x10 = self.at(ix+1, iy+0)
        x01 = self.at(ix+0, iy+1)
        x11 = self.at(ix+1, iy+1)

        def lerp(a, b, x):
            """Simple 1D linear interpolation formula"""
            return a*(1-x) + b*x

        return lerp(lerp(x00, x10, fx), lerp(x01, x11, fx), fy)


class BaselineFluidSolver:
    """ FluidSolver implements a semi-lagrangian method for solving
    incompressible fluid equations.

    This baseline implementation includes a single thread Gauss-Seidel
    for solving the pressure equation.
    """

    def __init__(self, size, fluid_density, fluid_quantities):
        """
        fluid_quantities is a list of quantities to advect
        """
        _dx = 1 / min(size[0], size[1])
        # Problem Constants
        self._w = size[0]
        self._h = size[1]
        self._dx = _dx
        self._rho = fluid_density

        # Fluid quantities
        # _u, _v are the x, y velocities on the border of a grid
        self._quantites = fluid_quantities
        self._u = FluidQuantity(size=(self._w+1, self._h),
                                offset=(0.0, 0.5), dx=_dx)
        self._v = FluidQuantity(size=(self._w, self._h+1),
                                offset=(0.5, 0.0), dx=_dx)

        # Buffers for solving fluid equation
        self._neg_div = np.zeros(self._w*self._h, dtype=np.float64)
        self._pressure = np.zeros(self._w*self._h, dtype=np.float64)

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
        coefficient = 1 / self._dx

        i = 0
        for iy in range(self._h):
            for ix in range(self._w):
                self._neg_div[i] = -coefficient * (self._u.at(ix+1, iy) - self._u.at(ix, iy) +
                                                   self._v.at(ix, iy+1) - self._v.at(ix, iy))
                i += 1

    def project(self, timestep, tol=1e-5, max_iter=600):
        """Eq. (4.19) Calculate pressure from divergence

        The pressure equation is a standard Laplacian, conjugate gradient is the method
        of choice for SPD matrices.

        This methods implements a Gauss-Seidel solver, mostly for ease of implementation
        """
        coefficient = timestep / (self._rho * self._dx * self._dx)

        max_error = 0
        for cur_iter in range(max_iter):
            max_error = 0
            for iy in range(self._h):
                for ix in range(self._w):
                    idx = ix + iy*self._w
                    diag, off_diag = 0, 0

                    # Boundary checking
                    if ix > 0:
                        diag += coefficient
                        off_diag -= coefficient * self._pressure[idx-1]
                    if iy > 0:
                        diag += coefficient
                        off_diag -= coefficient * self._pressure[idx-self._w]
                    if ix + 1 < self._w:
                        diag += coefficient
                        off_diag -= coefficient * self._pressure[idx+1]
                    if iy + 1 < self._h:
                        diag += coefficient
                        off_diag -= coefficient * self._pressure[idx-self._w]

                    new_pressure = (self._neg_div[idx] - off_diag) / diag
                    max_error = max(max_error, abs(
                        self._pressure - new_pressure))

            if max_error < tol:
                print('Pressure solver converged in {} iterations, max. error is {}\n'.format(
                    cur_iter, max_error))
                return

        print('Exceeded max. iter. of {}, max. error is {}\n'.format(
            max_iter, max_error))

    def make_incompressible(self, timestep):
        """Equation (4.4) and (4.5)

        Pressure should be updated using project(), now we calculate fluid velocities,
        to make the fluid incompressible.
        """
        coefficient = timestep / (self._rho * self._dx)

        idx = 0
        for iy in range(self._h):
            for ix in range(self._w):
                # self in lambda is referencing the FluidSolver object
                self._u.apply(ix, iy, lambda x: x -
                              coefficient*self._pressure[idx])
                self._u.apply(ix+1, iy, lambda x: x +
                              coefficient*self._pressure[idx])
                self._u.apply(ix, iy, lambda x: x -
                              coefficient*self._pressure[idx])
                self._u.apply(ix, iy+1, lambda x: x +
                              coefficient*self._pressure[idx])
                idx += 1

        # Set boundary back to 0
        for iy in range(self._h):
            self._u.apply(0, iy, lambda x: 0)
        for ix in range(self._w):
            self._v.apply(ix, 0, lambda x: 0)

    def advect(self, timestep):
        """Calculate fluid quantity values at the next time step.
        """
        # Advect all quantities
        for _, key in enumerate(self._quantites):
            self._quantites[key].advect(timestep, self._u, self._v)
        # Advect u and v themselves
        self._u.advect(timestep, self._u, self._v)
        self._v.advect(timestep, self._u, self._v)

        # Swap _val[] and _buf[] to commit result of advection
        for _, key in enumerate(self._quantites):
            self._quantites[key].swap()
        self._u.swap()
        self._v.swap()


def main():
    """Runs simulation
    """
    SIZE_X, SIZE_Y = 128, 128
    FLUID_DENSITY = 0.1
    # Section 3.2 discusses how to set this in some detail. Related by CFL condition
    # Here is empirically, as long as it's small enough, it is ok.
    TIMESTEP = 0.005
    MAX_TIME = 8
    N_STEPS = TIMESTEP // MAX_TIME

    frame = np.zeros((SIZE_X, SIZE_Y), dtype=np.int8)

    fluid_quantities = {
        'particle_density': FluidQuantity(
            size=(SIZE_X, SIZE_Y), offset=(0.5, 0.5), dx=1/min(SIZE_X, SIZE_Y))
    }
    fluid_solver = BaselineFluidSolver(
        size=(SIZE_X, SIZE_Y), fluid_density=FLUID_DENSITY, fluid_quantities=fluid_quantities)


if __name__ == '__main__':
    main()
