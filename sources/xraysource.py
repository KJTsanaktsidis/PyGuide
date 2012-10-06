import abc

class XRaySource(object):
    """
    This class is the abstract base class for all X-ray sources. It spits out, when requested, a wavefunction
    in x and z representing a wave eminating from this source.
    Subclasses should keep in mind that the coordinate system is guaranteed to center the waveguide on x = 0 and z = 0
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, tilt):
        """
        This constructor should initialise the subclass to produce plane waves with an appropriate tilt
        """
        pass

    @abc.abstractmethod
    def get_wave(self):
        """
        This method should be overridden by subclass to get another wavefunction
        """
        raise NotImplementedError()
