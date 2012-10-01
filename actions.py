"""Actions.py: actions determined by command line
This function contains methods that get called by __main__ in response to what command line arguments were specified
"""
import functools

import modesolvers
import splitters
import numpy as np
import scipy.io as sio
import csv
import os
import errno
import plotting

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import matplotlib


def ms_execute_permode(f, fbase, modes=(), kx=(), verbose=False):
    """
    This method executes a function f once for each mode specified in modes.
    f is executed with arguments f(mode, fname) as arguments

    @param f: the function to execute
    @type f: function
    @param fbase: The base of a filename passed in full into f
    @type fbase: str
    @param modes: The modes to iterate over or empty for all of them
    @type modes: list
    @param kx: The kx values of each mode
    @type kx: list
    @return: None
    """

    if len(modes) != 1:
        try:
            os.makedirs(fbase)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

    if len(modes) == 0:
        modes = range(1, len(kx) + 1)

    for mode in modes:
        if verbose:
            print 'Plotting n=%i' % mode
        if len(modes) == 1:
            fname = fbase
        else:
            fname = os.path.join(fbase, 'mode-%i.png' % mode)
        f(mode, fname)

def ms_plot_wavefunctions(waveguide, wavelength, out_file, verbose=False, modes=()):
    """
    This method plots the wavefunctions of the guided modes given by the Modes parameter

    @param waveguide: The waveguide whose guided modes to plot
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating this waveguide, in m
    @type wavelength: float
    @param verbose: Whether to give detailed output
    @type verbose: bool
    @param modes: A list of guided mode numbers to plot
    @type modes: list
    @return None
    """

    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kxs = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kxs, verbose=verbose)

    def f(mode, fname):
        wavef = lambda x: wavefs[mode-1](x, 0)
        (fig, reax, imax) = plotting.setup_figure_topbottom(title=u'Wavefunction for n=%i mode' % mode,
            xlabel=u'Distance accross waveguide (m)',
            ylabel=u'Wavefunction (arbitrary units)')
        plotting.plot_wavefunction(reax, imax, wavef, waveguide.slab_gap)
        plotting.save_figure(fig, fname)

    ms_execute_permode(f, out_file, modes, kxs, verbose=verbose)

def ms_plot_intensities(waveguide, wavelength, out_file, verbose=False, modes=(), dists=()):
    """
    This method plots the intensities wavefunctions of the guided modes given by the Modes parameter
    It is plotted at each of the distances given in the dists list, or at z=0 if empty

    @param waveguide: The waveguide whose guided modes to plot
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating this waveguide, in m
    @type wavelength: float
    @param verbose: Whether to give detailed output
    @type verbose: bool
    @param modes: A list of guided mode numbers to plot
    @type modes: list
    @return None
    """

    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kxs = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kxs, verbose=verbose)

    if len(dists) == 0:
        dists = [0]

    def f(mode, fname):
        wavefs_propagated = []
        labels = []

        for d in dists:
            wavefs_propagated.append(functools.partial(lambda z,x: wavefs[mode-1](x,z), d))
            labels.append('z=%.3fm' % d)

        (fig, ax) = plotting.setup_figure_standard(title=u'Intensity for n=%i mode' % mode,
            xlabel=u'Distance accross waveguide (m)',
            ylabel=u'Wavefunction intensity (arbitrary units)')
        #fig.hold(True)

        colours = ['blue', 'orange', 'green', 'purple', 'grey', 'lime', 'cyan', 'yellow', 'black', 'navy', 'teal']
        colours.reverse()
        ax.hold(True)
        for (wf, lbl) in zip(wavefs_propagated, labels):
            plotting.plot_intensity(ax, wf, waveguide.slab_gap, colour=colours.pop(), label=lbl)
        ax.hold(False)

        if len(dists) != 1:
            ax.legend(loc='upper left', prop={'size' : 10})

        plotting.save_figure(fig, fname)

    ms_execute_permode(f, out_file, modes, kxs, verbose=verbose)

def ms_plot_poynting(waveguide, wavelength, out_file, verbose=False, modes=()):
    """
    This method plots the poynting vector of the mode wavefunctions given by modes

    @param waveguide: The waveguide whose guided modes to plot
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating this waveguide, in m
    @type wavelength: float
    @param verbose: Whether to give detailed output
    @type verbose: bool
    @param modes: A list of guided mode numbers to plot
    @type modes: list
    @return None
    """
    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kxs = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kxs, verbose=verbose)

    def f(mode, fname):
        wf = functools.partial(lambda z,x: wavefs[mode-1](x,z), 0)
        (fig, ax) = plotting.setup_figure_standard(title=u'Poynting vector for n=%i mode' % mode,
            xlabel=u'Distance accross waveguide (m)',
            ylabel=u'Energy flow (right is +)')
        plotting.plot_poynting_vector(ax, wf, waveguide.slab_gap)
        plotting.save_figure(fig, fname)
    ms_execute_permode(f, out_file, modes=modes, kx=kxs, verbose=verbose)

def ms_plot_argand(waveguide, wavelength, out_file, verbose=False, modes=()):
    """
    This method plots an argand diagram of the wavefunctions given by modes

    @param waveguide: The waveguide whose guided modes to plot
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating this waveguide, in m
    @type wavelength: float
    @param verbose: Whether to give detailed output
    @type verbose: bool
    @param modes: A list of guided mode numbers to plot
    @type modes: list
    @return None
    """
    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kxs = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kxs, verbose=verbose)

    def f(mode, fname):
        wf = functools.partial(lambda z, x: wavefs[mode - 1](x, z), 0)
        (fig, ax) = plotting.setup_figure_standard(title=u'Argand diagram for n=%i mode' % mode,
            xlabel=ur'$\mathrm{Re}\left((\psi\left(x\right)\right)$',
            ylabel=ur'$\mathrm{Im}\left((\psi\left(x\right)\right)$')
        plotting.plot_argand(ax, wf, waveguide.slab_gap)
        plotting.save_figure(fig, fname)
    ms_execute_permode(f, out_file, modes=modes, kx=kxs, verbose=verbose)


def sp_plot_wavefunction(waveguide, wavelength, angle, out_file, verbose=False):
    """
    This method produces a wavefunction hitting the waveguide at angle, and splits it into the guided modes of the
    waveguide. The resultant wavefunction is plotted
    @param waveguide: The waveguide being illuminated
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating the waveguide
    @type wavelength: float
    @param angle: The angle of the plane waves striking the waveguide, in radian
    @type angle: float
    @param out_file: The filename to write output to
    @type out_file: str
    @param verbose: Whether or not to give verbose output
    @type verbose: bool
    @return: None
    """

    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kxs = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kxs, verbose=verbose)

    k = 2*np.pi/wavelength
    inkx = k*np.sin(angle)
    inwave = lambda x: np.exp(1j*inkx*x)

    splitter = splitters.ModeSplitter(inwave, wavefs)
    cm = splitter.get_coupling_constants()
    wffull = splitter.get_wavefunction(cm)
    wf = functools.partial(lambda z,x: wffull(x,z), 0.1)

    if verbose:
        print 'Coupling coefficients: '
        for (i,c) in enumerate(cm):
            print '\t %i: Magnitude: %f, Phase: %f, Square: %f' % (i+1, np.abs(c), np.angle(c), np.abs(c)**2)

    (fig, reax, imax) = plotting.setup_figure_topbottom(title=ur'Wavefunction for incidence angle $\theta=%f$ rad' % angle,
        xlabel=u'Distance accross waveguide (m)',
        ylabel=u'Wavefunction (arbitrary units)')
    plotting.plot_wavefunction(reax, imax, wf, waveguide.slab_gap)
    plotting.save_figure(fig, unicode(out_file))

def sp_plot_mode_angles(waveguide, wavelength, out_file, verbose=False):
    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kxs = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kxs, verbose=verbose)

    coefftable = np.zeros((len(kxs),len(kxs)))


    def f(mode, fname):
        kx = kxs[mode-1]
        inwave = lambda x: np.exp(1j*kx*x)
        splitter = splitters.ModeSplitter(inwave, wavefs)
        cm = splitter.get_coupling_constants()
        wffull = splitter.get_wavefunction(cm)
        wf = lambda x: wffull(x, 1)

        (fig, reax, imax) = plotting.setup_figure_topbottom(
            title=ur'Wavefunction for incidence angle on n=%i mode' % mode,
            xlabel=u'Distance accross waveguide (m)',
            ylabel=u'Wavefunction (arbitrary units)')
        #(fig, ax) = plotting.setup_figure_standard()
        plotting.plot_wavefunction(reax, imax, wf, waveguide.slab_gap)
        #plotting.plot_intensity(ax, wf, waveguide.slab_gap)
        plotting.save_figure(fig, fname)

        coefftable[mode-1, :] = np.abs(cm)**2
    ms_execute_permode(f, out_file, kx=kxs, verbose=verbose)
    sio.savemat(out_file + '_coeffs', {'coeffs' : coefftable})



def modesolver_find_kx(waveguide, wavelength, verbose):
    """This function is responsible for implementing modesolver --wavevectors behaviour

    @param waveguide: The waveguide being operated on
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating the waveguide
    @type wavelength: float
    """
    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kx = solver.solve_transcendental(verbose=verbose)
    k = waveguide.wavevector_length_core(wavelength)
    ang = np.abs(np.arcsin(np.array(kx) / k))
    return kx, ang

def modesolver_output_kx(kx, ang, out_file):
    """This function is responsible for helping with modesolver --wavevectors behaviour
    It writes out the calculated kx and ang values to a text file (CSV)
    """
    wr = csv.writer(out_file)
    wr.writerow(['Wavevector x-component (inverse metres)', 'Grazing angle (radians)'])
    for i in range(0, len(kx)):
        wr.writerow([kx[i], ang[i]])
    return
