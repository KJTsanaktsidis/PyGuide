"""Module for finding the eigenmodes of a waveguide
"""

import functools
import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.io
import numpy.linalg as nplinalg
import sys
import util

class LossySolver(object):
    """Class for solving a waveguide, taking loss in the cladding into account

    The operations in this class are based on page 69 of my lab book. This class finds the eigenmodes of a PlanarWaveguide by allowing kx to be a complex quantity when solving the transcendental numerically. This leads to a phase gradient in the core of the waveguide and hence a flow of energy into the cladding, as is required to preserve the waveguide shape.

    Approximations are made that the refractive index in the core is 1. If this is not the case, bad things will happen.
    """

    def __init__(self, waveguide, wavelength):
        """Constructor for the LossySolver

        @param waveguide: The waveguide whose eigenmodes to solve
        @type waveguide: PlanarWaveguide
        @param wavelength: The wavelength of x-rays which is illuminating this object
        @type wavelength: float
        """

        #: :type: PlanarWaveguide
        self.waveguide = waveguide
        #: :type: float
        self.wavelength = wavelength

    def num_modes(self):
        """Get the number of modes in the waveguide we are wrapping

        This method computes the number of modes in this waveguide. This is done by considering the refractive
        index of the cladding to be real, and computing V/pi as given in Fuhse

        @return: The number of guided modes in this waveguide
        @rtype: int
        """

        V = np.sqrt(self.waveguide.core_index(self.wavelength).real**2 -self.waveguide.cladding_index(self.wavelength).real**2)
        V = V * self.waveguide.slab_gap * self.waveguide.wavevector_length_core(self.wavelength).real
        return np.ceil(V/np.pi)

    def coeff_matrix(self, kx):
        """Compute the 4x4 coefficient matrix for finding A,B,C,D
        """
        n = self.waveguide.cladding_index(self.wavelength)
        k = self.waveguide.wavevector_length_core(self.wavelength)
        d = self.waveguide.slab_gap
        gamma = lambda kx: np.sqrt(k ** 2 - kx ** 2 - n ** 2 * k ** 2) #not n.real this time

        #previous array
        #return np.array([[1e7,         -1e7,                        0,                          0],
        #                 [0,         gamma(kx),                 kx,                         0],
        #                 [0,         kx*np.tan(kx*d)-gamma(kx), gamma(kx)*np.tan(kx*d)+kx,  0],
        #                 [0,         1e7*kx*np.sin(kx*d),       1e7*kx*np.cos(kx*d),          -1e7*gamma(kx)]])
        return np.array([[1e7,          -1e7,               0,                  0],
                         [gamma(kx),    0,                  kx,                 0],
                         [0,            1e7*np.cos(kx*d),   -1e7*np.sin(kx*d),  -1e7],
                         [0,            kx*np.sin(kx*d),    kx*np.cos(kx*d),    -gamma(kx)]])

    def reduced_coeff_matrix(self,kx):
        """Compute the 2x2 coefficient matrix for finding B,C
        """
        n = self.waveguide.cladding_index(self.wavelength)
        k = self.waveguide.wavevector_length_core(self.wavelength)
        d = self.waveguide.slab_gap
        gamma = lambda kx: np.sqrt(k ** 2 - kx ** 2 - n ** 2 * k ** 2) #not n.real this time

        return np.array([[gamma(kx),                        kx],
                         [kx * np.tan(kx * d) - gamma(kx),   gamma(kx) * np.tan(kx * d) + kx]])

    def get_wavefunctions(self, kx_array, verbose=False):
        """This method solves the matrix in coeff_matrix to give the four coefficients A, B, C and D
        It returns a list of functions representing the eigenfunctions of this waveguide
        """
        n = self.waveguide.cladding_index(self.wavelength)
        k = self.waveguide.wavevector_length_core(self.wavelength)
        d = self.waveguide.slab_gap
        gamma = lambda kx: np.sqrt(k ** 2 - kx ** 2 - n ** 2 * k ** 2) #not n.real this time


        wavefunctions = []
        for kx in kx_array:
            mat = self.coeff_matrix(kx)
            null_space = util.null_vector(mat)
            A = null_space[0]
            B = null_space[1]
            C = null_space[2]
            D = null_space[3]

            m = self.coeff_matrix(kx)
            ns = util.null_vector(m)
            #B = ns[0]
            #C = ns[1]
            #A = B
            #D = (kx*np.sin(kx*d)*B+kx*np.cos(kx*d)*C)/gamma(kx)

            kz = np.sqrt(k**2 - kx**2)

            def get_wavef_x(A, B, C, D, kx):
                def wavef_x(x):
                    if x > 0:
                        return A*np.exp(-gamma(kx)*x)
                    elif x < 0 and -d < x:
                        return B*np.cos(kx*x)+C*np.sin(kx*x)
                    else:
                        return D*np.exp(gamma(kx)*(x+d))
                return wavef_x

            #Normalise the area under the wavef
            wavef_x = get_wavef_x(A, B, C, D, kx)
            intensity = lambda x: np.abs(wavef_x(x))**2
            if verbose:
                print 'Integrating wavefunction kx = %s...' % kx
            ires = integrate.quad(intensity, -self.waveguide.slab_gap*10, self.waveguide.slab_gap*10,
                epsabs=1e-10, epsrel=1e-10, limit=400)
            area = ires[0]

            def get_wavef(wfx, ar, kz):
                return lambda x,z: wfx(x) * np.exp(1j*kz*z) / np.sqrt(ar)

            wavefunctions.append(get_wavef(get_wavef_x(A, B, C, D, kx), area, kz))

        return wavefunctions




    def solve_transcendental(self, verbose=True, singular_eps=1e-8):
        """Solve the transcendental equation in tan that yields kx of each mode

        In this implementation, kx is allowed to be complex. The complex index of refraction of the cladding material is explicitely used, so the tangent on the LHS can be complex too- yielding complex kx.

        The algorithm is to:
            -First consider only real n
                -Compute the critical angle
                -Compute the number of modes we would expect for this waveguide
                -Splatter 10*this number of starting points accross the interval [0 critical angle]
                -Solve the transcendental starting at each of these points
                -Collate the solutions
            -Then, consider commplex n:
                -Take the solutions for the lossless wsaveguide
                -Solve the transcendental using each of these solutions as an initial guess

        @return: A list of x-wavevector components that represent the guided modes of this waveguide
        @rtype: list
        """

        n = self.waveguide.cladding_index(self.wavelength)
        k = self.waveguide.wavevector_length_core(self.wavelength).real
        d = self.waveguide.slab_gap

        if verbose:
            delta = 1-n.real
            beta = n.imag
            print 'Using refractive index: n = 1 - %s + %si' % (delta, beta)

        #We thankfully know exactly where the functions are going to be undefined
        #We can therefore bracket the roots on them
        undefined_kx = []
        #The RHS function will be undefined for kx^2>n^2k*2-k^2
        func_stop_kx = np.sqrt(-n.real ** 2 * k ** 2 + k**2)
        undefined_kx.append(func_stop_kx)
        #the RHS function will be undefined for 2*kx^2==k^2-k^2*n^2
        undefined_kx.append(np.sqrt(0.5*(k**2 - n.real**2 * k**2)))
        #LHS will be undefined for every kx=pi/2d*n, n odd
        undefined_kx += list(np.arange(np.pi/(2*d), func_stop_kx, np.pi/d))

        #get our undef points in order
        undefined_kx.sort()
        if verbose:
            print 'List of undefined points: '
            for kx in undefined_kx:
                print '\tkx = %f' % kx

        #now make brakcets out of it
        brackets = []
        padding = func_stop_kx * 1e-5
        for i in range(0, len(undefined_kx)-1): #iterate over every element but last
            #add a bit of padding, since we need the function to actually be evaluatable at the brackets
            bracket_min = undefined_kx[i] + padding
            bracket_max = undefined_kx[i+1] - padding
            brackets.append([bracket_min, bracket_max])

        if verbose:
            print 'Using brackets:'
            for bracket in brackets:
                print '\t(%f, %f)' % (bracket[0], bracket[1])

        #Now we can do root finding on each of our brackets
        real_kx = []
        gamma = lambda kx: np.sqrt(k**2 - kx**2 - n.real**2 * k**2)
        f_to_zero = lambda kx: np.tan(kx*d) - (2*kx*gamma(kx))/(kx**2 - gamma(kx)**2)
        #Our choice of brackets begins at the first tan asymptote. Therefore, the
        for bracket in brackets:
            #try rootfinding
            try:
                (kx, r) = optimize.brentq(f_to_zero, bracket[0], bracket[1], full_output=True)
            except ValueError:
                continue
            if r.converged:
                if verbose:
                    print 'Found real-case solution kx = %f' % (kx)
                real_kx.append(kx)
            else:
                if verbose:
                    print 'Could not converge on bracket (%f, %f)!' % (bracket[0], bracket[1])
            sys.stdout.flush()

        #Did we get the right number of roots?
        N = self.num_modes()
        if N > len(real_kx):
            raise TranscendentalSolveError("Insufficient number of roots found (found %i, expected %i)" %
                                            (len(real_kx), N))
        elif N < len(real_kx):
            raise TranscendentalSolveError("Too many roots found (found %i, expected %i)" %
                                            (len(real_kx), N))
        elif verbose:
            print 'Successfully found real-case roots'

        #If all they wanted was the real solution, throw it to them now
        if n == n.real:
            return real_kx

        #Now use vector optimisation, with real_kx as a starting point
        gamma = lambda kx: np.sqrt(k ** 2 - kx ** 2 - n** 2 * k ** 2) #not n.real this time
        def vector_f_to_zero(x):
            #complex kx
            kx = x[0] + x[1]*1j
            cf = f_to_zero(kx) #defined above
            return [cf.real, cf.imag]

        def vector_f_to_min(x):
            return -np.log10(nplinalg.cond(self.coeff_matrix(x[0]+x[1]*1j)))

        complex_kx = []
        i = 1
        for rkx in real_kx:
            #try and minimise function starting from hereabouts
            root_result = optimize.root(vector_f_to_zero, [rkx, -rkx*1e-3], method='lm',
                options={'xtol':1e-150, 'ftol':1e-150})
            root_kx = root_result['x'][0]+root_result['x'][1]*1j
            if verbose:
                print 'Using kx=%s as the complex minimisation starting point for real root kx=%s' % (root_kx, rkx)
                condition_number = nplinalg.cond(self.coeff_matrix(root_kx))
                print 'Condition number: %s' % condition_number
                sys.stdout.flush()

            #now use root_result as the starting point for an exhaustive global minimisation in the viscinity
            bracket = [(root_result['x'][0]-10, root_result['x'][0]+10),
                       (root_result['x'][1]-10, root_result['x'][1]+10)]
            #min_result = optimize.fmin_l_bfgs_b(vector_f_to_min, root_result['x'],fprime=None, approx_grad=True,
            #    bounds=bracket, m=1000, factr=0.001, pgtol=1e-15, epsilon=1e-150, maxfun=45000)
            min_result = optimize.minimize(vector_f_to_min, root_result.x, method='Nelder-Mead', #bounds=bracket,
                options={'xtol' : 1e-150, 'ftol' : 1e-150, 'maxfev' : 90000})
            optim_kx = min_result.x[0] + min_result.x[1]*1j
            condition_number = nplinalg.cond(self.coeff_matrix(optim_kx))
            if verbose:
                print 'Optimised to kx=%s; condition number %s' % (optim_kx, condition_number)
                sys.stdout.flush()

            i = i + 1
            complex_kx.append(optim_kx)

            #raise TranscendentalSolveError('Could not converge on a complex solution for kx = %f (found kx = %s)'
            #    % (rkx, optim_kx))

        sys.stdout.flush()
        return complex_kx

class ExpLossySolver(LossySolver):

    def coeff_matrix(self, kx):
        """Compute the 4x4 coefficient matrix for finding A,B,C,D
        """
        n = self.waveguide.cladding_index(self.wavelength)
        k = self.waveguide.wavevector_length_core(self.wavelength)
        d = self.waveguide.slab_gap
        gamma = lambda kx: np.sqrt(k ** 2 - kx ** 2 - n ** 2 * k ** 2) #not n.real this time
        g = gamma(kx)
        return np.matrix([[-1,       np.exp(1j*kx*d/2),          np.exp(-1j*kx*d/2),        0],
                         [0,        np.exp(-1j*kx*d/2),         np.exp(1j*kx*d/2),        -1],
                         [g,        1j*kx* np.exp(1j*kx*d/2),   -1j*kx*np.exp(-1j*kx*d/2), 0],
                         [0,        1j*kx*np.exp(-1j*kx*d/2),   -1j*kx*np.exp(1j*kx*d/2),  -g]])

    def get_wavefunctions(self, kx_array, verbose=False):
        """This method solves the matrix in coeff_matrix to give the four coefficients A, B, C and D
        It returns a list of functions representing the eigenfunctions of this waveguide
        """
        n = self.waveguide.cladding_index(self.wavelength)
        k = self.waveguide.wavevector_length_core(self.wavelength)
        d = self.waveguide.slab_gap
        gamma = lambda kx: np.sqrt(k ** 2 - kx ** 2 - n ** 2 * k ** 2) #not n.real this time

        nmode = 0
        wavefunctions = []
        mats = dict()
        for kx in kx_array:
            nmode += 1
            mat = self.coeff_matrix(kx)
            mats['m'+str(nmode)] = mat
            nullv = util.null_vector(mat)
            A = nullv[0]
            B = nullv[1]
            C = nullv[2]
            D = nullv[3]

            if verbose:
                solnerrormat = np.matrix(self.coeff_matrix(kx))*np.matrix([[A],[B],[C],[D]])
                print 'n = %i Solution Error: %f' % (nmode, nplinalg.norm(solnerrormat))
                print 'Components: %s' % np.abs(solnerrormat)

            kz = np.sqrt(k**2 - kx**2)
            def wavef(x, z, A, B, C, D, kx, kz, g, mult):
                if x > d/2:
                    xpart = A*np.exp(-g*(x-d/2))
                elif x >= -d/2:
                    xpart = B*np.exp(1j*kx*x) + C*np.exp(-1j*kx*x)
                else:
                    xpart = D*np.exp(g*(x+d/2))
                return xpart*np.exp(1j*kz*z)*mult

            wavef_captured = functools.partial(wavef, A=A, B=B, C=C, D=D, kx=kx, kz=kz, g=gamma(kx), mult=1)
            wavef_x = functools.partial(wavef_captured, z=0)
            intensity = lambda x: np.abs(wavef_x(x))**2
            ires = integrate.quad(intensity, -self.waveguide.slab_gap * 10, self.waveguide.slab_gap * 10,
                epsabs=1e-10, epsrel=1e-10, limit=400)
            area = ires[0]
            wavef_norm = functools.partial(wavef, A=A, B=B, C=C, D=D, kx=kx, kz=kz, g=gamma(kx), mult=1/np.sqrt(area))
            wavefunctions.append(wavef_norm)
        return wavefunctions





class TranscendentalSolveError(Exception):
    pass
