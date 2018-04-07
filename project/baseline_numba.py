"""
This work is based off of Bridson's "Fluid Simulation", SIGGRAPH 2007 Notes.
https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf
"""

import time

import numpy as np
from numba import float64, jit, uint32, void
from PIL import Image

from utils import timeSince


@jit(void(float64[:],
          float64, float64, float64, float64, float64,
          float64[:], float64[:],
          float64))
def _set_block(v, vw, vh, vdx, vox, voy, top_left, bot_right, value):
    """Sets the fluid quantity within top_left and bot_right rectangle
    to have value returned by f.

    top_left: (x0, y0)
    bot_right: (x1, y1)
    f: ()
    """
    ix0 = int(top_left[0]/vdx - vox)
    iy0 = int(top_left[1]/vdx - voy)
    ix1 = int(bot_right[0]/vdx - vox)
    iy1 = int(bot_right[1]/vdx - voy)

    for iy in range(max(iy0, 0), min(iy1, vh)):
        for ix in range(max(ix0, 0), min(ix1, vw)):
            v[ix+iy*vw] = value


@jit(void(float64[:],
          float64[:], uint32,
          float64[:], uint32,
          uint32, uint32, float64), nopython=True)
def _calc_neg_div(neg_div, u, uw, v, vw, w, h, dx):
    """Eq. (4.11) and (4.13)
    This construction is trivially parallizable, no reason to do it in a loop
    """
    coefficient = 1 / dx

    idx = 0
    for iy in range(h):
        for ix in range(w):
            neg_div[idx] = -coefficient * (
                u[(ix+1)+iy*uw] - u[ix+iy*uw] +
                v[ix+(iy+1)*vw] - v[ix+iy*vw])
            idx += 1


@jit(float64[:](float64[:], float64[:],
                uint32, uint32, float64,
                float64, float64,
                float64, uint32), nopython=True)
def _project(pressure, neg_div, w, h, dx, rho, timestep, tol=1e-5, max_iter=600):
    """Eq. (4.19) Calculate pressure from divergence

    The pressure equation is a standard Laplacian, conjugate gradient is the method
    of choice for SPD matrices.

    This methods implements a Gauss-Seidel solver, mostly for ease of implementation
    """
    coefficient = timestep / (rho * dx * dx)
    cur_iter = 0
    max_error = 0

    for cur_iter in range(max_iter):
        max_error = 0
        for iy in range(h):
            for ix in range(w):
                idx = ix + iy*w
                diag, off_diag = 0, 0

                # Boundary checking
                if ix > 0:
                    diag += coefficient
                    off_diag -= coefficient * pressure[idx-1]
                if iy > 0:
                    diag += coefficient
                    off_diag -= coefficient * pressure[idx-w]
                if ix < w - 1:
                    diag += coefficient
                    off_diag -= coefficient * pressure[idx+1]
                if iy < h - 1:
                    diag += coefficient
                    off_diag -= coefficient * pressure[idx+w]

                new_pressure = (neg_div[idx] - off_diag) / diag
                max_error = max(max_error, abs(
                    pressure[idx] - new_pressure))
                pressure[idx] = new_pressure

        if max_error < tol:
            return np.array([cur_iter, max_error])

    return np.array([max_iter, max_error])


@jit(void(float64[:], uint32,
          float64[:], uint32,
          float64[:],
          uint32, uint32, float64,
          float64, float64), nopython=True)
def _make_incompressible(u, uw, v, vw, pressure, w, h, dx, rho, timestep):
    """Equation (4.4) and (4.5)

    Pressure should be updated using project(), now we calculate fluid velocities,
    to make the fluid incompressible.
    """
    coefficient = timestep / (rho * dx)

    idx = 0
    for iy in range(h):
        for ix in range(w):
            # self in lambda is referencing the FluidSolver object
            diff = coefficient * pressure[idx]
            u[ix+iy*uw] -= diff
            u[(ix+1)+iy*uw] += diff
            v[ix+iy*vw] -= diff
            v[ix+(iy+1)*vw] += diff
            idx += 1

    # Set boundary back to 0
    for iy in range(h):
        u[0+iy*uw] = 0
        u[w+iy*uw] = 0
    for ix in range(w):
        v[ix+0*vw] = 0
        v[ix+h*vw] = 0


@jit(float64(float64, float64, float64))
def two_point_interpolate(a, b, x):
    """Simple linear interpolation"""
    return a*(1-x) + b*x


@jit(float64(float64, float64,
             float64[:], uint32, uint32, float64, float64))
def _mac_grid_interpolate(x, y, w, ww, wh, wox, woy):
    """2D linear interpolate the quantity values at (x, y) from four
    near by grid points.
    """
    # Boundary checking
    x = min(max(x - wox, 0.0), ww - 1.001)
    y = min(max(y - woy, 0.0), wh - 1.001)
    # Extract integer and fractional parts
    ix = int(x)
    iy = int(y)
    x -= ix
    y -= iy

    x00 = w[(ix+0)+(iy+0)*ww]
    x10 = w[(ix+1)+(iy+0)*ww]
    x01 = w[(ix+0)+(iy+1)*ww]
    x11 = w[(ix+1)+(iy+1)*ww]

    return two_point_interpolate(
        two_point_interpolate(x00, x10, x),
        two_point_interpolate(x01, x11, x), y)


@jit(float64(float64[:], float64[:],
             uint32, uint32, float64, float64, float64,
             float64,
             float64[:], uint32, uint32, float64, float64,
             float64[:], uint32, uint32, float64, float64))
def _advect(w, wbuf,
            ww, wh, wox, woy, wdx,
            timestep,
            u, uw, uh, uox, uoy,
            v, vw, vh, vox, voy):
    """Advect the fluid quantity use velocity fields of u, v

    Eq. (3.6) to (3.9).  This is the Lagrangian part of the
    semi-Lagrangian methods.
    """
    idx = 0
    for iy in range(wh):
        for ix in range(ww):
            x = ix + wox
            y = iy + woy

            # Forward Euler method
            # Divide by dx because we want an index value
            x -= _mac_grid_interpolate(x, y, u, uw,
                                       uh, uox, uoy) * timestep / wdx
            y -= _mac_grid_interpolate(x, y, v, vw,
                                       vh, vox, voy) * timestep / wdx

            # We now know the value at the next time step should be
            # the old value at (x, y), however this might not be perfectly
            # on a grid point, so we interpolate again.
            # Writing to buf[] because we still need to use the old values
            # to advect other quantities.
            wbuf[idx] = _mac_grid_interpolate(x, y, w, ww, wh, wox, woy)
            idx += 1


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
        """Calls kernel function and writes to self._buf
        """
        _advect(self._val, self._buf,
                self._w, self._h, self._ox, self._oy, self._dx,
                timestep,
                u._val, u._w, u._h, u._ox, u._oy,
                v._val, v._w, v._h, v._ox, v._oy)

    def linear_interpolate(self, x, y):
        """2D linear interpolate the quantity values at (x, y) from four
        near by grid points.
        """
        # Boundary checking
        x = min(max(x - self._ox, 0.0), self._w - 1.001)
        y = min(max(y - self._oy, 0.0), self._h - 1.001)
        # Extract integer and fractional parts
        ix = (int)(x)
        iy = (int)(y)
        x -= ix
        y -= iy

        x00 = self.at(ix+0, iy+0)
        x10 = self.at(ix+1, iy+0)
        x01 = self.at(ix+0, iy+1)
        x11 = self.at(ix+1, iy+1)

        def lerp(a, b, x):
            """Simple 1D linear interpolation formula"""
            return a*(1-x) + b*x

        return lerp(lerp(x00, x10, x), lerp(x01, x11, x), y)


class BaselineFluidSolver:
    """ FluidSolver implements a semi-lagrangian method for solving
    incompressible fluid equations.

    This baseline implementation includes a single thread Gauss-Seidel
    for solving the pressure equation.
    """

    def __init__(self, size, fluid_density):
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
        self._p = FluidQuantity(size=(self._w, self._h),
                                offset=(0.5, 0.5), dx=_dx)
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
        """ Docs in kernel function"""
        _calc_neg_div(self._neg_div,
                      self._u._val, self._u._w,
                      self._v._val, self._v._w,
                      self._w, self._h, self._dx)

    def project(self, timestep, tol=1e-5, max_iter=600):
        """ Docs in kernel function"""
        iters, error = _project(self._pressure, self._neg_div, self._w, self._h,
                                self._dx, self._rho, timestep, tol, max_iter)
        print('Project finished in %d iterations, final error is %g' %
              (iters, error))

    def make_incompressible(self, timestep):
        """ Docs in kernel function"""
        _make_incompressible(self._u._val, self._u._w,
                             self._v._val, self._v._w,
                             self._pressure,
                             self._w, self._h, self._dx,
                             self._rho, timestep)

    def advect(self, timestep):
        """Calculate fluid quantity values at the next time step.
        """
        # Advect all quantities
        self._p.advect(timestep, self._u, self._v)
        # Advect u and v themselves
        self._u.advect(timestep, self._u, self._v)
        self._v.advect(timestep, self._u, self._v)

        # Swap _val[] and _buf[] to commit result of advection
        self._p.swap()
        self._u.swap()
        self._v.swap()

    def set_condition(self):
        """Sets the condition of the simulation environment.
        """

        # Source 1
        _set_block(self._p._val,
                   self._p._w, self._p._h, self._p._dx, self._p._ox, self._p._oy,
                   [0.20, 0.2], [0.30, 0.21], 1.0)

        _set_block(self._u._val,
                   self._u._w, self._u._h, self._u._dx, self._u._ox, self._u._oy,
                   [0.20, 0.2], [0.30, 0.21], 0.0)

        _set_block(self._v._val,
                   self._v._w, self._v._h, self._v._dx, self._v._ox, self._v._oy,
                   [0.20, 0.2], [0.30, 0.21], 3.0)

        # Source 2
        _set_block(self._p._val,
                   self._p._w, self._p._h, self._p._dx, self._p._ox, self._p._oy,
                   [0.70, 0.2], [0.80, 0.21], 1.0)

        _set_block(self._u._val,
                   self._u._w, self._u._h, self._u._dx, self._u._ox, self._u._oy,
                   [0.70, 0.2], [0.80, 0.21], 0.0)

        _set_block(self._v._val,
                   self._v._w, self._v._h, self._v._dx, self._v._ox, self._v._oy,
                   [0.70, 0.2], [0.80, 0.21], 3.0)


def main():
    """Runs simulation
    """
    SIZE_X, SIZE_Y = 128, 128
    FLUID_DENSITY = 0.1
    # Section 3.2 discusses how to set this in some detail. Related by CFL condition
    # Here is empirically, as long as it's small enough, it is ok.
    TIMESTEP = 0.005
    MAX_TIME = 8
    N_STEPS = (int)(MAX_TIME/TIMESTEP)
    PRINT_EVERY = 4

    # A buffer that stores a RGB PNG image to be outputted
    pixels = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8)

    fluid_solver = BaselineFluidSolver(
        size=(SIZE_X, SIZE_Y), fluid_density=FLUID_DENSITY)

    def print_image(particle_density, image, filename):
        """Outputs a PNG using particle_density FluidQuantity"""

        size = image.shape[0:2]
        shade = ((1 - particle_density._val.reshape(size))
                 * 255.0).astype('uint8')
        image[:, :, 0] = shade
        image[:, :, 1] = shade
        image[:, :, 2] = shade

        im = Image.fromarray(image)
        im.save(filename, 'PNG', quality=100)

    img_index = 0
    start_time = time.time()
    for step in range(1, N_STEPS+1):
        fluid_solver.set_condition()
        fluid_solver.step(TIMESTEP)

        if step % PRINT_EVERY == 0:
            print_image(fluid_solver._p,
                        pixels, 'output/frame{:05d}.png'.format(img_index))
            img_index += 1

        print('%s (%d %d%%)' % (timeSince(start_time, step / N_STEPS),
                                step, step / N_STEPS * 100))


if __name__ == '__main__':
    main()
