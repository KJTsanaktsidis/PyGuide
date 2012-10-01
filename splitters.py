import scipy.integrate as integrate
import functools
import numpy as np

class ModeSplitter(object):
    """
    This class performs the computation of splitting a plane wave of a certain incident angle
    represented by the inwave argument to the constructor into guided modes via an overlap integral.
    it can be used to retrieve a wavefunction representing the flow of this wave in the waveguide
    at arbitrary distances z0 downstream of z=0.
    """
    def __init__(self, inwave, modewaves):
        """
        Constructor for the modesplitter
        @param inwave: Wavefunction as a function of x of the incident wave when it hits the waveguide
        @type inwave: function
        @param modewaves: An ordered array of wavefunctions for each guided mode in the waveguide
        @type modewaves: list
        @return None
        """
        self.inwave = inwave
        self.modewaves = modewaves

    def get_coupling_constants(self):
        """
        This method returns the coupling constants obtained by evaluating the overlap integral
        They are normalised so that the sum of their squares is 1
        @return: The coupling constants obtained by evaluating the overlap integral
        @rtype list
        """
        inwaveconj = lambda x: self.inwave(x).conjugate()
        inwaveintensity = lambda x: np.abs(inwaveconj(x))**2
        #Need to normalise the inwave
        #factor = integrate.quad(inwaveintensity, -1e-6, 1e-6)[0]
        #inwaveconj = lambda x: self.inwave(x).conjugate()/np.sqrt(factor)


        cm = []
        sum = 0
        for (n, wavef) in enumerate(self.modewaves):
            wavefclip = functools.partial(lambda z,x: wavef(x,z), 0)
            integrand = lambda x: inwaveconj(x)*wavef(x, 0)
            reint = lambda x: integrand(x).real
            imint = lambda x: integrand(x).imag
            reval = integrate.quad(reint, -1e-6, 1e-6, limit=500)[0]
            imval = integrate.quad(imint, -1e-6, 1e-6, limit=500)[0]
            sum += np.abs(reval+1j*imval)**2
            cm.append(reval+1j*imval)
        for (n,c) in enumerate(cm):
            cm[n] = c/np.sqrt(sum)
        return cm

    def get_wavefunction(self, cms):
        """
        This method constructs and returns a wavefunction that represents the propagation of inwave
        through the waveguide
        @param cms: The coupling coefficients
        @type cms: list
        @return: A wavefunction
        @rtype function
        """
        def f(x,z, mult=1):
            val = 0
            for (cm, wavef) in zip(cms, self.modewaves):
                val += cm*wavef(x,z)
            return val*mult
        f0 = lambda x: f(x, 0)
        integrand = lambda x: np.abs(f0(x))**2
        area = integrate.quad(integrand, -1e-6, 1e-6)[0]
        fnorm = lambda x,z: f(x,z, mult=1/np.sqrt(area))
        #return fnorm
        return f