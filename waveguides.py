"""PyGuide2.waveguides: A module for storing different types of waveguides

This module contains various classes containing different kinds of waveguides
"""

import abc
import numpy as np
import xml.etree.ElementTree as ElementTree
import csv
import os

class Waveguide(object):
    """A base class for X-Ray waveguides
    All of the waveguides in PyGuide2.waveguides inherit from this class
    """
    __metaclass__ = abc.ABCMeta

class PlanarWaveguide(Waveguide):
    """Represents an infinite (in y-direction) planar x-ray waveguide

    It contains a core material and a cladding material, along with dimensions and information about the type of light that it is being illuminated with.
    """

    def __init__(self, core_index = 1, cladding_index = 1, slab_gap = 1e-7, length=0.05):
        """Constructor for the PlanarWaveguide

        Properties can be specified on construction by using kwargs
        Note that since the refractive indicies are specified as a single number, this ties this object to a single incident
        wavelength (since refractive index depends on wavelength)
        This might be changed in a subclass that overrides core_index() and cladding_index() to do something more intelligent

        @param core_index: The refractive index of the core material
        @type core_index: complex
        @param cladding_index: The refractive index of the cladding material
        @type cladding_index: complex
        @param slab_gap: The distance (in metres) between the two slabs in the waveguide
        @type slab_gap: float
        @param length: The axial length of the waveguide (in meres)
        @type length: float
        """

        #: :type: complex
        self.i_core_index = core_index
        #: :type: complex
        self.i_cladding_index = cladding_index
        #: :type: float
        self.slab_gap = slab_gap
        #: :type: float
        self.length = length

    def core_index(self, wavelength): #This method is reserved for overloading to compute index as a function of wavelength
        return self.i_core_index
    def cladding_index(self, wavelength):
        return self.i_cladding_index

    def wavevector_length_core(self, wavelength):
        """Gets the length of the wavevector in the core of this waveguide

        @param wavelength: The wavelength of light that is illuminating this object
        @type wavelength: float
        @rtype: complex
        """
        wl = wavelength / self.core_index(wavelength)
        return 2*np.pi/wl

    def wavevector_length_cladding(self, wavelength):
        """Gets the length of the wavevector in the cladding of this waveguide

        @param wavelength: The wavelength of light that is illuminating this object
        @type wavelength: float
        @rtype: complex
        """

        wl = wavelength / self.cladding_index(wavelength)
        return 2*np.pi/wl

    def critical_angle(self, wavelength):
        """
        This method returns the critical angle for this waveguide
        Note that since it uses the self.cladding_index() method, it will work if this class is later overridden
        to take into account n as a function of wavelength

        @param wavelength: The wavelength of light that is illuminating this object
        @type wavelength: float
        @return: The critical angle
        @rtype: float
        """
        return np.sqrt(2*(1-self.cladding_index(wavelength).real)) #1-n is delta

    @staticmethod
    def get_waveguide(key):
        """Get a canned waveguide

        This method pulls a waveguide specified by key from the known_waveguides dictionary.
        @param key: The key representing the waveguide to get
        @type key: string
        @rtype: PlanarWaveguide
        """
        #This is a set of pre-fabb'd waveguides
        known_waveguides = {'losswg': PlanarWaveguide(core_index=1,
        cladding_index=1 - 7.67e-6 + 1.77e-7j,
        slab_gap=100e-9, length=0.05),
                        'perfectwg': PlanarWaveguide(core_index=1,
                            cladding_index=1 - 7.67e-6,
                            slab_gap=100e-9, length=0.05), }

        return known_waveguides[key]

class PlanarWaveguideInterp(PlanarWaveguide):
    """This class extends PlanarWaveguide to account for n as a function of wavelength.

    It contains code for loading waveguides from xml files and interpolating delta and beta to get values of n
    as a function of wavelength
    """

    def __init__(self, xmldesc):
        """Constructor for PlanarWaveguideInterp that takes an XML file describing a waveguide
        The XML file is parsed and is used to fill up this class

        @param xmldesc: The xml file describing this waveguide
        @type xmldesc: file
        """
        if xmldesc is None:
            super(PlanarWaveguideInterp, self).__init__()
            self.interp_cladding_index = np.array([0, 0, 0])
            return

        tree = ElementTree.parse(xmldesc)
        #All of these should be defined
        slab_gap = float(tree.find('slabgap').text)
        length = float(tree.find('length').text)
        i_core_index = complex(tree.find('coreindex').text)

        super(PlanarWaveguideInterp, self).__init__(core_index=i_core_index, slab_gap=slab_gap, length=length)

        cladding_tree = tree.find('claddingindex')
        if 'type' in cladding_tree.attrib and cladding_tree.attrib['type'] == 'text/csv':
            cladding_csv = cladding_tree.text
            cladding_iter = iter(cladding_csv.splitlines())

            format = csv.Sniffer().sniff(cladding_csv)
            format.skipinitialspace = True


            reader = csv.reader(cladding_iter, format)
            rows = []
            for row in reader:
                if len(row) < 3:
                    continue
                #Col 1 is wavelength, 2 is delta, 3 is beta
                r = [0,0,0]
                r[0] = float(row[0])*1e-9 #wavelength is given in nm from CXRO
                r[1] = float(row[1])
                r[2] = float(row[2])
                rows.append(r)

            self.cladding_data = np.array(rows)

            #Make an interpolating function
            #Zero is a problem
            self.cladding_data[:,1:] = self.cladding_data[:,1:] + 1e-20
            linearised_data = np.log(self.cladding_data)
            self.coeff_linear_vector = np.polyfit(linearised_data[:,0], linearised_data[:,1:], 1)
        else:
            self.i_cladding_index = complex(cladding_tree.text)


    def cladding_index(self, wavelength):
        """This method gives the refractive index in the cladding
        It works by applying the results of the interpolation done in __init__
        """
        try:
            logx = np.log(wavelength)
            logd = self.coeff_linear_vector[0, 0] * logx + self.coeff_linear_vector[1, 0]
            logb = self.coeff_linear_vector[0, 1] * logx + self.coeff_linear_vector[1, 1]
            return 1 - (np.exp(logd)-1e-20) + (np.exp(logb)-1e-20) * 1j
        except NameError:
            return self.i_cladding_index #not doing anything fancy with wavelength

    @staticmethod
    def get_waveguide(key):
        path = os.path.dirname(__file__)
        xmlpath = os.path.join(path, 'waveguides', key + '.xml')
        return PlanarWaveguideInterp(open(xmlpath))