"""Actions.py: actions determined by command line
This function contains methods that get called by __main__ in response to what command line arguments were specified
"""
import functools

import modesolvers
import numpy as np
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
