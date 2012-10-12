from matplotlib.figure import Figure
from matplotlib.figure import Axes
from matplotlib.text import Text
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from numpy import linspace, vectorize, min, max, abs, meshgrid
from scipy.misc import derivative
import matplotlib.cm as cm

def setup_figure_standard(title='', xlabel='', ylabel=''):
    """
    This method sets up a figure and axis object for plotting a matplotlib figure with the correct aesthetics
    for PyGuide

    @param title: The title of the plot
    @type title: unicode
    @param xlabel: Label for the x-axis of the plot
    @type xlabel: unicode
    @param ylabel: Label for the y-axis of the plot
    @type ylabel: unicode
    @return: A tuple of (figure, axis) for this figure
    @rtype tuple
    """

    fig = Figure(figsize=(5.5, 4.25))
    #: :type: Axes
    ax = fig.add_subplot(111)

    ax.set_title(title, size=12)
    ax.set_xlabel(xlabel, size=10)
    ax.set_ylabel(ylabel, size=10)
    ax.ticklabel_format(scilimits=(-3,3))

    return (fig, ax)

def setup_figure_topbottom(title='', xlabel='', ylabel=''):
    """
    This method sets up a figure with two subplots for plotting, eg, a function with real and imag parts
    The axes are placed top to bottom.
    THe title is placed above the top figure, the xlabel below the bottom one, and the ylabel on the side of the two

    @param title: The title of the plot
    @type title: unicode
    @param xlabel: Label for the x-axis of the plot
    @type xlabel: unicode
    @param ylabel: Label for the y-axis of the plot
    @type ylabel: unicode
    @return: A tuple of (figure, topaxis, bottomaxis) for this figure
    @rtype tuple
    """

    fig = Figure(figsize=(5.5, 4.25))
    #: :type: Axes
    topax = fig.add_subplot(211)
    #: :type: Axes
    bottomax = fig.add_subplot(212, sharex=topax)

    topax.set_title(title, size=12)
    bottomax.set_xlabel(xlabel, size=10)
    bottomax.set_ylabel(ylabel, size=10)

    #place the ylabel in the middle of the two plots
    bpos = bottomax.get_yaxis().get_label().get_position()
    bottomax.get_yaxis().get_label().set_position((bpos[0], 1))

    topax.ticklabel_format(scilimits=(-3, 3))
    bottomax.ticklabel_format(scilimits=(-3, 3))

    #fig.subplots_adjust()
    return (fig, topax, bottomax)

def shade_waveguide(ax, slabGap):
    """
    This method adds shading to indicate the core and cladding regions of the waveguide in ax

    @param ax: The axes to add teh shading to
    @type ax: Axes
    @param slabGap: The slab gap of the planar waveguide
    @type slabGap: float
    @return: None
    """

    ystart = min(ax.get_ylim())
    yend = max(ax.get_ylim())
    ysize = yend - ystart

    leftr = Rectangle((-slabGap, ystart), slabGap/2, ysize, hatch='/', fill=False)
    rightr = Rectangle((slabGap/2, ystart), slabGap/2, ysize, hatch='\\', fill=False )

    ax.add_patch(leftr)
    ax.add_patch(rightr)


def save_figure(fig, filename):
    """
    This method prepares a figure for being saved, and then writes it to the file given by filename

    @param fig: The figure to save
    @type fig: Figure
    @param filename: The name of the file to write to
    @type filename: unicode
    @return: None
    """

    #change the font size on the tick labels
    for ax in fig.get_axes():
        for ticklabel in ax.get_xticklabels():
            ticklabel.set_size(8)
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_size(8)
        ax.get_xaxis().get_offset_text().set_size(8)
        ax.get_yaxis().get_offset_text().set_size(8)
        #change size of legend if applicable
        if ax.get_legend() is not None:
            for label in ax.get_legend().get_texts():
                label.set_size(10)

    fig.subplots_adjust()
    #fig.tight_layout()
    c = FigureCanvas(fig)
    c.print_figure(filename, dpi=300)

def plot_wavefunction(reax, imax, wf, slabGap, colours=('blue', 'green')):
    """
    This method plots the real part of the wavefunction to reax and the imaginary part to imax
    It is plotted between -slabGap and slabGap

    @param reax: The axes to write the real part of the wavefunction to
    @type reax: Axes
    @param imax: The axes to write the imaginary part of the wavefunction to
    @type imax: Axes
    @param wf: The wavefunction to plot, as a function of x
    @type wf: function
    @param slabGap: The slab gap of the planar waveguide (used for setting the plotting bounds)
    @type slabGap: float
    @param colours: A tuple of colours to plot the (real, imaginary) parts in
    @type colours: tuple
    @return: None
    """

    ref = lambda x: wf(x).real
    imf = lambda x: wf(x).imag
    revf = vectorize(ref)
    imvf = vectorize(imf)

    #evaluate functions
    x = linspace(-slabGap, slabGap, 2000)
    rey = revf(x)
    imy = imvf(x)

    #do the plotting
    reax.plot(x, rey, linestyle='-', color=colours[0], label=r'$\mathrm{Re}\left(\psi\left(x\right)\right)$',
        antialiased=True, marker=None)
    imax.plot(x, imy, linestyle='-', color=colours[1], label=r'$\mathrm{Im}\left(\psi\left(x\right)\right)$',
        antialiased=True, marker=None)

    #we need labels for this too
    reax.legend(loc='upper left', prop={'size' : 10})
    #imax.legend(loc='upper left', prop={'size' : 10})

    reax.set_xlim((-slabGap, slabGap))
    imax.set_xlim((-slabGap, slabGap))
    reax.set_ylim((min(rey)-0.1*abs(min(rey)), max(rey)+0.1*abs(max(rey))))
    imax.set_ylim((min(imy) - 0.1 * abs(min(imy)), max(imy) + 0.1 * abs(max(imy))))

    shade_waveguide(reax, slabGap)
    shade_waveguide(imax, slabGap)

def plot_intensity(ax, wf, slabGap, colour='blue', label=''):
    """
    This method plots the intensity of a wavefunction to ax
    It is plotted between -slabGap and slabGap

    @param ax: The axes to write the intensity of the wavefunction to
    @type ax: Axes
    @param wf: The wavefunction to plot, as a function of x
    @type wf: function
    @param slabGap: The slab gap of the planar waveguide (used for setting the plotting bounds)
    @type slabGap: float
    @param colours: A colour recognised by matplotlib to plot in
    @type colours: object
    @return: None
    """

    intensity = lambda x: abs(wf(x))**2
    x = linspace(-slabGap, slabGap, 2000)
    vf = vectorize(intensity)
    y = vf(x)

    ax.plot(x, y, linestyle='-', color=colour, label=label, antialiased=True, marker=None)

    #only change the axes limits if we havn't seen this object before
    if not hasattr(ax, 'pyguide_afiemw_haveseen'):
        ax.pyguide_afiemw_haveseen = True
        ax.set_xlim((-slabGap, slabGap))
        ax.set_ylim((min(y) - 0.1 * abs(min(y)), max(y) + 0.1 * abs(max(y))))
        shade_waveguide(ax, slabGap)

def plot_poynting_vector(ax, wf, slabGap, colour='blue'):
    """
    This method plots the poynting vector of wf to ax, using a stable algorithm that does not involve
    differentiating the phase of wf

    @param ax: The axes to plot to
    @type ax: Axes
    @param wf: The wavefunction whose poynting vector to plot, as a function of x
    @type wf: function
    @param slabGap: The slab gap of the planar waveguide
    @type slabGap: float
    @param colour: The colour to plot on
    @type colour: object
    @return: None
    """

    wfconj = lambda x: wf(x).conjugate()
    dwf = lambda x: derivative(wf, x, dx=1e-11)
    poynting = lambda x: (wfconj(x)*dwf(x)).imag

    vf = vectorize(poynting)
    x = linspace(-slabGap, slabGap, 2000)
    y = vf(x)

    ax.plot(x, y, linestyle='-', color=colour, antialiased=True, marker=None)
    ax.set_xlim((-slabGap, slabGap))
    ax.set_ylim((min(y) - 0.1 * abs(min(y)), max(y) + 0.1 * abs(max(y))))
    shade_waveguide(ax, slabGap)

def plot_intensity_map(ax, wf, slabGap, z):
    """
    This method plots a 2-d map of intensity accross and down the waveguide.

    @param ax: The axes to plot to
    @type ax: Axes
    @param wf: The wavefunction whose intensity to plot, as a function of x and z
    @type wf: function
    @param slabGap: The slab gap of the planar waveguide
    @type slabGap: float
    @param z: A tuple representing the (start, end) z planes to plot between
    @type z: tuple
    @return: None
    """

    intensity = lambda x,z: abs(wf(x,z))**2
    (X,Y) = meshgrid(linspace(z[0], z[1], 500), linspace(-slabGap, slabGap, 500))
    vf = vectorize(intensity)
    Z = vf(Y,X)

    pcm = ax.pcolor(X, Y, Z, cmap=cm.jet)
    ax.get_figure().colorbar(pcm, ax=ax, use_gridspec=True)
    ax.set_axes([z[0], z[1], -slabGap, slabGap])

    #bang down a slightly different guideline for the waveguide limits
    topr = Rectangle((z[0], slabGap/2), z[1]-z[0], slabGap/2, hatch='\\', fill=False)
    bottomr = Rectangle((z[0], -slabGap), z[1]-z[0], slabGap/2, hatch='/', fill=False)
    ax.add_patch(topr)
    ax.add_patch(bottomr)



def plot_argand(ax, wf, slabGap):
    tvec = linspace(-slabGap, slabGap, 2000)
    xdata = []
    ydata = []
    for t in tvec:
        xdata.append(wf(t).real)
        ydata.append(wf(t).imag)

    ax.plot(xdata, ydata, linestyle='-', antialiased=True, marker=None)
