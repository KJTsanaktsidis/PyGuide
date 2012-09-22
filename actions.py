"""Actions.py: actions determined by command line
This function contains methods that get called by __main__ in response to what command line arguments were specified
"""

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
            wavefs_propagated.append(lambda x: wavefs[mode-1](x, d))
            labels.append('z=%fm' % d)

        (fig, ax) = plotting.setup_figure_standard(title=u'Intensity for n=%i mode' % mode,
            xlabel=u'Distance accross waveguide (m)',
            ylabel=u'Wavefunction intensity (arbitrary units)')

        colours = ['blue', 'orange', 'green', 'yellow', 'navy', 'lime', 'cyan', 'purple', 'black', 'grey', 'teal']
        for (wf, lbl) in zip(wavefs_propagated, labels):
            plotting.plot_intensity(ax, wf, waveguide.slab_gap, colour=colours.pop(), label=lbl)

        if len(dists) != 0:
            ax.legend(loc='upper right')

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

def modesolver_output_attncoeffs(k, kx, ang, out_file):
    """This function is responsible for implementing modesolver --attncoeffs behaviour
    It takes the kx values (and angles) of waveguide modes and computes the
    effective linear attenuation coefficient, writing it to a file

    @param k: The wavevector of the incident light
    @type k: float
    @param kx: The x-component wavevectors of the waveguide mode
    @type kx: list
    @param ang: The incidence angles of the waveguide modes
    @type ang: list
    @param out_file: The file to output to
    @type out_file: file
    """
    kzs = np.sqrt(k**2 - np.power(kx,2))
    lacs = kzs.imag
    wr = csv.writer(out_file)
    wr.writerow(['Wavevector x-component (inverse metres)', 'Grazing angle (radians)',
                 'Effective linear attenuation coefficient'])
    for i in range(0, len(kx)):
        wr.writerow([kx[i], ang[i], lacs[i]])
    return

def modesolver_plot_loss(modes, waveguide, wavelength, distances, out_file, verbose=False):
    """This function is responsible for implementing modesolver --lossplot behaviour
    It takes the same arguments as modesolver_plot_modes, as well as the distances at which to plot
    the intensity

    @param modes: The modes for which to print plots
    @type modes: list
    @param waveguide: The waveguide whose modes to print
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating the waveguide
    @type wavelength: float
    @param distances: The distances (in m) away from the waveguide enterance to plot intensity
    @type distances: list
    @param out_file: The file or directory in which to place the output
    @type out_file: unicode
    @param verbose: Whether or not to give verbose output
    @type verbose: bool
    """

    solver = modesolvers.LossySolver(waveguide, wavelength)
    kx = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kx, verbose=verbose)

    if len(modes) > 1:
        try:
            os.makedirs(out_file)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

    for m in modes:
        wavef = wavefs[m-1]
        intensity = lambda x,z: np.abs(wavef(x,z))**2

        fig = Figure(figsize=(5.5, 4.25))
        canvas = FigureCanvas(fig)
        ax = fig.add_axes((0.23, 0.1, 0.73, 0.8))

        x = np.linspace(-2 * waveguide.slab_gap, waveguide.slab_gap, num=300)

        colours = ['blue', 'orange', 'green', 'yellow','navy', 'lime', 'cyan', 'purple', 'black', 'grey', 'teal']
        colours.reverse()
        maxy = 0
        ax.hold(True)
        fig.hold(True)
        for dist in distances:
            vf = np.vectorize(intensity)
            y = vf(x, dist)
            if np.max(y) > maxy:
                maxy = np.max(y)
            ax.plot(x, y, color=colours.pop(), linestyle='-', marker=None, antialiased=True,
                label='d=%.3fm' % dist)

        ax.hold(False)
        fig.hold(False)

        ax.legend(loc='upper left', prop={'size' : 8})

        ax.set_xlim((-2 * waveguide.slab_gap, waveguide.slab_gap))
        ax.set_ylim((0, maxy+0.1*maxy))
        ax.set_xlabel('Distance accross waveguide (m)')
        ax.set_ylabel(r'Wavefunction Intensity $|\psi(x)|^2$ (arbitrary)')
        ax.set_title('Wavefunction of n=%i guided mode' % m)

        #whack on red lines for the waveguide extent
        l1 = Line2D([-waveguide.slab_gap, -waveguide.slab_gap], [0, 1e8], linestyle='-', color='red', marker=None)
        l2 = Line2D([0, 0], [0, 1e8], linestyle='-', color='red', marker=None)
        ax.add_line(l1)
        ax.add_line(l2)

        if len(modes) == 1:
            #f = open(out_file, 'w')
            fname = out_file
        else:
            fname = os.path.join(out_file, 'mode-%i.png' % m)
            #f = open(fname, 'w')
        canvas.print_figure(fname, dpi=300, edgecolor='white')


def modesolver_plot_wavefunctions(modes, waveguide, wavelength, out_file, verbose=False):
    """This function is responsible for implementing modesolver --wavefunctions behaviour
    It plots the wavefunctions specified by modes (both real and imag parts) at z=0

    @param modes: The modes for which to print plots
    @type modes: list
    @param waveguide: The waveguide whose modes to print
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating the waveguide
    @type wavelength: float
    @param out_file: The file or directory in which to place the output
    @type out_file: unicode
    @param verbose: Whether or not to give verbose output
    @type verbose: bool
    """

    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kx = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kx, verbose=verbose)

    if len(modes) > 1:
        try:
            os.makedirs(out_file)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

    matplotlib.rcParams.update({'font.size': 8})
    for m in modes:
        wavef = wavefs[m-1]
        fig = Figure(figsize=(5.5, 4.25)) #size in inches
        canvas = FigureCanvas(fig)
        #ax = fig.add_axes((0.23, 0.1, 0.73, 0.8)) #0.25in padding
        reax = fig.add_subplot(211)
        imax = fig.add_subplot(212)

        realf = lambda x: wavef(x,0).real
        imagf = lambda x: wavef(x,0).imag
        realvf = np.vectorize(realf)
        imagvf = np.vectorize(imagf)
        x = np.linspace(-waveguide.slab_gap, waveguide.slab_gap, num=300)
        ry = realvf(x)
        iy = imagvf(x)

        reax.plot(x, ry, color='blue', linestyle='-', label=r'$\mathrm{Re}\left(\psi\left(x\right)\right)$',
            marker=None, antialiased=True)
        imax.plot(x, iy, color='green', linestyle='-', label=r'$\mathrm{Im}\left(\psi\left(x\right)\right)$',
            marker=None, antialiased=True)

        reax.legend(loc='upper left', prop={'size': 8})
        imax.legend(loc='upper left', prop={'size': 8})
        reax.set_xlim((-waveguide.slab_gap, waveguide.slab_gap))
        reax.set_ylim((1.1*np.min(ry), 1.1*np.max(ry)))
        reax.set_xlabel('Distance accross waveguide (m)')
        reax.set_ylabel(r'Wavefunction Component (arbitrary)')
        reax.set_title('Wavefunction of n=%i guided mode' % m)
        imax.set_xlim((-waveguide.slab_gap, waveguide.slab_gap))
        imax.set_ylim((1.1*np.min(iy), 1.1*np.max(iy)))
        imax.set_xlabel('Distance accross waveguide (m)')
        imax.set_ylabel(r'Wavefunction Component (arbitrary)')

        #whack on red lines for the waveguide extent
        l1 = Line2D([-waveguide.slab_gap/2, -waveguide.slab_gap/2], [-1e10, 1e10], linestyle='-', color='red', marker=None)
        l2 = Line2D([waveguide.slab_gap/2, waveguide.slab_gap/2], [-1e10, 1e10], linestyle='-', color='red', marker=None)
        reax.add_line(l1)
        reax.add_line(l2)
        imax.add_line(l1)
        imax.add_line(l2)

        if len(modes) == 1:
            #f = open(out_file, 'w')
            fname = out_file
        else:
            fname = os.path.join(out_file, 'mode-%i.png' % m)
            #f = open(fname, 'w')

        canvas.print_figure(fname, dpi=300, edgecolor='white')

def modesolver_plot_modes(modes, waveguide, wavelength, out_file, verbose=False):
    """This function is responsible for implementing modesolver --modeplot
    It takes as an argument a list of guided modes to print, and the waveguide properties,
    and plots the modes at z=0 into out_file
    If multiple plots are required, out_file is created as a directory
    Plot is printed in bounds -2d - d

    @param modes: The modes for which to print plots
    @type modes: list
    @param waveguide: The waveguide whose modes to print
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating the waveguide
    @type wavelength: float
    @param out_file: The file or directory in which to place the output
    @type out_file: unicode
    @param verbose: Whether or not to give verbose output
    @type verbose: bool
    """

    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kx = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kx, verbose=verbose)

    if len(modes) > 1:
        try:
            os.makedirs(out_file)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

    for m in modes:
        wavef = wavefs[m-1] #modes are specified 1-based

        fig = Figure(figsize=(5.5, 4.25)) #size in inches
        canvas = FigureCanvas(fig)
        ax = fig.add_axes((0.23, 0.1, 0.73, 0.8)) #0.25in padding

        intensity = lambda x: np.abs(wavef(x,0))**2
        vf = np.vectorize(intensity)
        x = np.linspace(-2*waveguide.slab_gap, waveguide.slab_gap, num=300)
        y = vf(x)

        ax.plot(x, y, color='b', linestyle='-', marker=None, antialiased=True)
        ax.set_xlim((-2*waveguide.slab_gap, waveguide.slab_gap))
        ax.set_ylim((-1.1*np.abs(np.min(y)), np.max(y)+0.1*np.max(y)))
        ax.set_xlabel('Distance accross waveguide (m)')
        ax.set_ylabel(r'Wavefunction Intensity $|\psi(x)|^2$ (arbitrary)')
        ax.set_title('Wavefunction of n=%i guided mode' % m)

        #whack on red lines for the waveguide extent
        l1 = Line2D([-waveguide.slab_gap, -waveguide.slab_gap], [-1e10, 1e10], linestyle='-', color='red', marker=None)
        l2 = Line2D([0, 0], [-1e10, 1e10], linestyle='-', color='red', marker=None)
        ax.add_line(l1)
        ax.add_line(l2)

        if len(modes) == 1:
            #f = open(out_file, 'w')
            fname = out_file
        else:
            fname = os.path.join(out_file, 'mode-%i.png' % m)
            #f = open(fname, 'w')

        canvas.print_figure(fname, dpi=300, edgecolor='white')

def modesolver_plot_phase(modes, waveguide, wavelength, out_file, verbose=False):
    """This method is responsible for implementing modesolver --modephaseplots behaviour
    It finds the wavefunctions in the waveguide for the guided modes specified, and plots
    their phase gradient as a function of waveguide profile

    @param modes: The guided modes to produce plots for
    @type modes: list
    @param waveguide: The waveguide to analyse
    @type waveguide: PlanarWaveguide
    @param wavelength: The wavelength of light illuminating the waveguide
    @type wavelength: float
    @param out_file: The directory or file to write output to
    @type out_file: unicode
    @param verbose: Whether or not to produce verbose output
    @type verbose: bool
    """
    solver = modesolvers.ExpLossySolver(waveguide, wavelength)
    kx = solver.solve_transcendental(verbose=verbose)
    wavefs = solver.get_wavefunctions(kx, verbose=verbose)

    if len(modes) > 1:
        try:
            os.makedirs(out_file)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

    for m in modes:
        wavef = wavefs[m - 1] #modes are specified 1-based

        fig = Figure(figsize=(5.5, 4.25)) #size in inches
        canvas = FigureCanvas(fig)
        ax = fig.add_axes((0.23, 0.1, 0.73, 0.8)) #0.25in padding

        #phase = lambda x: np.log((wavef(x,0)/np.abs(wavef(x, 0) ** 1))).imag
        phase = lambda x: np.angle(wavef(x,0))
        vf = np.vectorize(phase)
        x = np.linspace(-waveguide.slab_gap, waveguide.slab_gap, num=300)
        xstep = x[2] - x[1]
        phasearr = vf(x)
        y = np.diff(phasearr, n=1) / xstep #compute numerical derivative of phasearray
        #x = x[:-1] #omit last element of x since y shrunk during differntiation
        y = phasearr

        ax.plot(x, y, color='b', linestyle='-', marker=None, antialiased=True)
        ax.set_xlim((-2 * waveguide.slab_gap, waveguide.slab_gap))
        ax.set_ylim((np.min(y) + 0.1*np.min(y), np.max(y) + 0.1 * np.max(y)))
        ax.set_xlabel('Distance accross waveguide (m)')
        ax.set_ylabel(r'Wavefunction Phase Gradient (arbitrary)')
        ax.set_title('Wavefunction Phase Gradient of n=%i Guided Mode' % m)

        #whack on red lines for the waveguide extent
        l1 = Line2D([-waveguide.slab_gap/2, -waveguide.slab_gap/2], [-1e10, 1e10], linestyle='-', color='red', marker=None)
        l2 = Line2D([waveguide.slab_gap / 2, waveguide.slab_gap / 2], [-1e10, 1e10], linestyle='-', color='red', marker=None)
        ax.add_line(l1)
        ax.add_line(l2)

        if len(modes) == 1:
            #f = open(out_file, 'w')
            fname = out_file
        else:
            fname = os.path.join(out_file, 'mode-%i.png' % m)
            #f = open(fname, 'w')

        canvas.print_figure(fname, dpi=300, edgecolor='white')
